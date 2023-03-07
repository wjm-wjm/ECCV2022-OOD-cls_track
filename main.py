import argparse
import easydict
import os
import sys
import yaml
import random
import shutil
import time
import warnings
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed


from utils import Logger, save_checkpoint, mkdir_if_missing, get_macs, get_num_parameters
from model import CreateModel
from dataloading import data_loading
from optim import select_optimizer_lr, adjust_learning_rate
from engine import trainer, tester, tester_save_csv, tester_final_save_csv
from loss import MaeContrastiveLoss


def main(args_main, args_config):
    if not args_main.evaluate:
        if not args_main.multiprocessing_distributed:
            sys.stdout = Logger(os.path.join(args_config.log_dir, "log_train_os.txt"))
            test_os_log = open(os.path.join(args_config.log_dir, "log_test_os.txt"), "w")
        else:
            sys.stdout = Logger(os.path.join(args_config.log_dir, "log_train_os.txt"))
            test_os_log = None
        mkdir_if_missing(os.path.join(args_config.log_dir, "images"))
    else:
        if not args_main.multiprocessing_distributed:
            sys.stdout = Logger(os.path.join(args_config.log_dir, "log_test.txt"))
        test_os_log = None

    print("==========\nargs_main:{}\n==========".format(args_main))
    print("==========\nargs_config:{}\n==========".format(args_config))

    if args_config.seed:
        random.seed(args_config.seed)
        np.random.seed(args_config.seed)
        torch.manual_seed(args_config.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. "
                    + "This will turn on the CUDNN deterministic setting, "
                    + "which can slow down your training considerably! "
                    + "You may see unexpected behavior when restarting "
                    + "from checkpoints.")

    if args_main.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args_main.dist_url == "env://" and args_main.world_size == -1:
        args_main.world_size = int(os.environ["WORLD_SIZE"])

    args_main.distributed = args_main.world_size > 1 or args_main.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args_main.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args_main.world_size = ngpus_per_node * args_main.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args_main, args_config, test_os_log))
    else:
        # Simply call main_worker function
        main_worker(args_main.gpu, ngpus_per_node, args_main, args_config, test_os_log)


def main_worker(gpu, ngpus_per_node, args_main, args_config, test_os_log):
    args_main.gpu = gpu

    if args_main.gpu is not None:
        print("Use GPU: {} for training".format(args_main.gpu))

    if args_main.distributed:
        if args_main.dist_url == "env://" and args_main.rank == -1:
            args_main.rank = int(os.environ["RANK"])
        if args_main.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args_main.rank = args_main.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args_main.dist_backend, init_method=args_main.dist_url, world_size=args_main.world_size, rank=args_main.rank)

    model = CreateModel(args_config, args_config.arch, class_num=10)

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif args_main.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args_main.gpu is not None:
            torch.cuda.set_device(args_main.gpu)
            model.cuda(args_main.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args_config.batch_size = int(args_config.batch_size / ngpus_per_node)
            args_config.workers = int((args_config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args_main.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args_main.gpu is not None:
        torch.cuda.set_device(args_main.gpu)
        model = model.cuda(args_main.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args_config.arch.startswith("alexnet") or args_config.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    n_parameters = get_num_parameters(model)
    print("Number of Params: {}M ({})".format(n_parameters * 1e-6, n_parameters))
    MACs = get_macs(model, (args_config.input_size, args_config.input_size))
    print("MACs: {}G ({})".format(MACs * 1e-9, MACs))

    end = time.time()
    train_loader, val_loader_iid, val_loader_nui, val_loader_final, class_num, class_name, train_sampler = data_loading(args_main, args_config)
    print("time for preparing training and validation dataloader: {}s({}h)".format(time.time() - end, (time.time() - end)/3600))

    # define optimizer
    optimizer, scheduler = select_optimizer_lr(args_config, model)

    # optionally resume from a checkpoint
    if args_main.resume:
        if os.path.isfile(args_main.resume):
            # print("=> loading checkpoint '{}'".format(args_main.resume))
            if args_main.gpu is None:
                checkpoint = torch.load(args_main.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args_main.gpu)
                checkpoint = torch.load(args_main.resume, map_location=loc)
            args_config.start_epoch = checkpoint["epoch"]
            best_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("=> loaded checkpoint '{}' (epoch {})".format(args_main.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args_main.resume))

    cudnn.benchmark = True

    if args_config.cutmix:
        print("=> using cutmix augmentation")
    if args_config.patch_aug:
        print("=> using patch augmentation")

    if args_main.evaluate:
        tester_final_save_csv(args_main, args_config, class_name, model, val_loader_final)
        # tester_save_csv(args_main, args_config, class_name, model, val_loader_iid, val_loader_nui)
        acc1_all_iid, acc5_all_iid, acc1_all_nui_avg, acc5_all_nui_avg = tester(args_main, args_config, args_config.start_epoch, model, val_loader_iid, val_loader_nui)
        print("* Best Epoch: [{}]  All(iid)  Acc@1: {:.2f}  All(nuisances-avg)  Acc@1: {:.2f}\n".format(best_epoch, acc1_all_iid, acc1_all_nui_avg))

        return

    best_acc1_all_nui_avg = 0
    best_acc1_all_iid = 0
    best_epoch = 0
    threshold = 91.1
    training_time = 0
    validation_time = 0
    for epoch in range(args_config.start_epoch, args_config.epochs):
        if args_main.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        end = time.time()
        trainer(args_main, args_config, epoch, optimizer, model, train_loader)
        training_time += (time.time() - end)
        print("time for training per epoch: {}s({}h)".format(time.time() - end, (time.time() - end) / 3600))

        # evaluate on validation set
        end = time.time()
        acc1_all_iid, acc5_all_iid, acc1_all_nui_avg, acc5_all_nui_avg = tester(args_main, args_config, epoch, model, val_loader_iid, val_loader_nui, test_os_log)
        validation_time += (time.time() - end)
        print("time for validation per epoch: {}s({}h)".format(time.time() - end, (time.time() - end) / 3600))

        scheduler.step()
        # lr = adjust_learning_rate(args_config, optimizer, epoch)
        # print("current lr: {}".format(lr))

        # remember best acc@1 and save checkpoint
        is_best = acc1_all_nui_avg > best_acc1_all_nui_avg
        if is_best and acc1_all_iid <= threshold:
            best_acc1_all_nui_avg = acc1_all_nui_avg
            best_acc1_all_iid = acc1_all_iid
            best_epoch = epoch
        print("* Best Epoch: [{}]  All(iid)  Acc@1: {:.2f}  All(nuisances-avg)  Acc@1: {:.2f}\n".format(best_epoch, best_acc1_all_iid, best_acc1_all_nui_avg))
        if test_os_log is not None:
            print("* Best Epoch: [{}]  All(iid)  Acc@1: {:.2f}  All(nuisances-avg)  Acc@1: {:.2f}\n".format(best_epoch, best_acc1_all_iid, best_acc1_all_nui_avg), file=test_os_log)

        if not args_main.multiprocessing_distributed or (args_main.multiprocessing_distributed and args_main.rank % ngpus_per_node == 0):
            save_checkpoint({
                "epoch": epoch,
                "arch": args_config.arch,
                "state_dict": model.state_dict(),
                "best_acc1_all_nui_avg": best_acc1_all_nui_avg,
                "best_acc1_all_iid": best_acc1_all_iid,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, (is_best and acc1_all_iid <= threshold), args_config.log_dir)

        if test_os_log is not None:
            test_os_log.flush()

    print("time for training: {}s({}h)".format(training_time, training_time / 3600))
    print("time for validation: {}s({}h)".format(validation_time, validation_time / 3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECCV2022 WorkShop OOD-cls")
    parser.add_argument("--config", help="path to config file", required=True)
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
    parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use")
    parser.add_argument("--multiprocessing-distributed", action="store_true", help="Use multi-processing distributed training to launch "
                                                                                   "N processes per node, which has N GPUs. This is the "
                                                                                   "fastest way to use PyTorch for either single node or "
                                                                                   "multi node data parallel training")

    args_main = parser.parse_args()
    args_config = yaml.load(open(args_main.config), Loader=yaml.FullLoader)
    args_config = easydict.EasyDict(args_config)

    main(args_main, args_config)

