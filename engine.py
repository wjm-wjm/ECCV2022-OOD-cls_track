import os
import time
import copy
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import pandas as pd
from utils import Summary, AverageMeter, ProgressMeter, mkdir_if_missing

from torchvision.utils import save_image
import torchvision.transforms as transforms

import ttach as tta
import kornia as K


def de_norm(x):
    """
    x size: [B, C, H, W]
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    out = (x.cpu() * std.view(-1, 1, 1) + mean.view(-1, 1, 1))
    return out.clamp(0, 1)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def patch_level_aug(input1, patch_transform, upper_limit, lower_limit):
    bs, channle_size, H, W = input1.shape
    patches = input1.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5).contiguous().reshape(-1,
                                                                                                        channle_size,
                                                                                                        16, 16)
    patches = patch_transform(patches)

    patches = patches.reshape(bs, -1, channle_size, 16, 16).permute(0, 2, 3, 4, 1).contiguous().reshape(bs,
                                                                                                        channle_size * 16 * 16,
                                                                                                        -1)
    output_images = F.fold(patches, (H, W), 16, stride=16)
    output_images = clamp(output_images.cuda(), lower_limit, upper_limit)
    return output_images


def accuracy(output, target, topk=(1,)):
    # Computes the accuracy over the k top predictions for the specified values of k (pytorch version)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res


def accuracy_numpy(output, target, topk=(1,)):
    # Computes the accuracy over the k top predictions for the specified values of k (numpy version)
    maxk = max(topk)
    data_size = output.shape[0]

    pred = np.argsort(-output)
    correct = (pred == target.reshape((-1, 1)))

    res = []
    for k in topk:
        correct_k = correct[:, :k].sum() * 1.0
        res.append(correct_k * 100.0 / data_size)

    return res


def trainer(args_main, args_config, epoch, optimizer, model, train_loader):
    batch_time = AverageMeter("Time", ":6.3f", summary_type=Summary.AVERAGE)
    data_time = AverageMeter("Data", ":6.3f", summary_type=Summary.AVERAGE)
    loss_total = AverageMeter("Loss(Total)", ":.4e", summary_type=Summary.AVERAGE)
    loss_ce = AverageMeter("Loss(CE)", ":.4e", summary_type=Summary.AVERAGE)
    loss_mae = AverageMeter("Loss(MAE)", ":.4e", summary_type=Summary.AVERAGE)
    loss_mnce = AverageMeter("Loss(MNCE)", ":.4e", summary_type=Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", summary_type=Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", summary_type=Summary.AVERAGE)
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_total, loss_ce, loss_mae, loss_mnce, top1, top5], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # if args_main.gpu is not None:
        #     images = images.cuda(args_main.gpu, non_blocking=True)
        # if torch.cuda.is_available():
        #     target = target.cuda(args_main.gpu, non_blocking=True)
        images = images.cuda()
        target = target.cuda()

        if (args_config.cutmix_prob <= 1.0) and (args_config.cutmix_prob > 0):
            r = np.random.rand(1)
            if r < args_config.cutmix_prob:
                args_config.cutmix = True
            else:
                args_config.cutmix = False

        if args_config.cutmix:
            # generate mixed sample
            lam = np.random.beta(args_config.beta, args_config.beta)
            rand_index = torch.randperm(images.size()[0])
            target_rand = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            # images_origin = copy.deepcopy(images)
            # images_origin[rand_index, :, bbx1:bbx2, bby1:bby2] = images_origin[:, :, bbx1:bbx2, bby1:bby2]
            # image_mask = torch.ones_like(images)
            # image_mask[:, :, bbx1:bbx2, bby1:bby2] = 0.0
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        if args_config.patch_aug:
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16, 16), scale=(0.85, 1.0), ratio=(1.0, 1.0), p=0.5),
                # K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.5),
                K.augmentation.RandomGrayscale(p=0.5),
                # K.augmentation.RandomHorizontalFlip(p=0.5)
            )
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
            upper_limit = ((1 - mu_imagenet) / std_imagenet).cuda()
            lower_limit = ((0 - mu_imagenet) / std_imagenet).cuda()
            images = patch_level_aug(images, patch_transform, upper_limit, lower_limit)

        # compute output and losses
        if args_config.arch == "resnet50":
            if args_config.cutmix:
                output, ce_loss = model(images, target, target_rand, lam)
            else:
                output, ce_loss = model(images, target)
            ce_loss = ce_loss.mean()

        if args_config.arch == "mae":
            if args_config.cutmix:
                output, mae_loss, ce_loss, mnce_loss, pred, imgs, imgs_mask = model(images, target, target_rand, lam)
            else:
                output, mae_loss, ce_loss, mnce_loss, pred, imgs, imgs_mask = model(images, target)
            mae_loss = mae_loss.mean()
            ce_loss = ce_loss.mean()
            mnce_loss = mnce_loss.mean()

        if args_config.arch == "resnet50 + mae":
            if args_config.cutmix:
                output, mae_loss, ce_loss, mnce_loss, pred, imgs, imgs_mask = model(images, target, target_rand, lam)
            else:
                output, mae_loss, ce_loss, mnce_loss, pred, imgs, imgs_mask = model(images, target)
            mae_loss = mae_loss.mean()
            ce_loss = ce_loss.mean()
            mnce_loss = mnce_loss.mean()

        # compute total loss
        if args_config.arch == "resnet50":
            total_loss = ce_loss

        if args_config.arch == "mae":
            total_loss = ce_loss + mae_loss + mnce_loss

        if args_config.arch == "resnet50 + mae":
            total_loss = ce_loss + mae_loss + mnce_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_ce.update(ce_loss.item(), images.size(0))
        if args_config.arch == "mae":
            loss_mae.update(mae_loss.item(), images.size(0))
            loss_mnce.update(mnce_loss.item(), images.size(0))

        if args_config.arch == "resnet50 + mae":
            loss_mae.update(mae_loss.item(), images.size(0))
            loss_mnce.update(mnce_loss.item(), images.size(0))
        loss_total.update(total_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args_config.print_freq == 0:
            if "mae" in args_config.arch:
                save_image(de_norm(pred), os.path.join(args_config.log_dir, "images", "pred_" + str(epoch) + "_" + str(i) + ".jpg"))
                save_image(de_norm(imgs), os.path.join(args_config.log_dir, "images", "imgs_" + str(epoch) + "_" + str(i) + ".jpg"))
                save_image(de_norm(imgs_mask), os.path.join(args_config.log_dir, "images", "imgs_mask_" + str(epoch) + "_" + str(i) + ".jpg"))
            progress.display(i)


def tester(args_main, args_config, epoch, model, val_loader_iid, val_loader_nui, test_os_log=None):
    val_loader = val_loader_iid + val_loader_nui
    acc1_all_iid = 0
    acc5_all_iid = 0
    acc1_all_nui_avg = 0
    acc5_all_nui_avg = 0

    def aug_transform():
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180]),
            ]
        )
        return transforms

    # switch to evaluate mode
    model.eval()
    # model = tta.ClassificationTTAWrapper(model, tta.aliases.ten_crop_transform(224, 224))

    for name, loader in val_loader:
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        progress = ProgressMeter(len(loader), [batch_time, top1, top5], prefix="Test_{}: ".format(name))

        with torch.no_grad():
            end = time.time()
            for i, (images, target, path) in enumerate(loader):
                # if args_main.gpu is not None:
                #     images = images.cuda(args_main.gpu, non_blocking=True)
                # if torch.cuda.is_available():
                #     target = target.cuda(args_main.gpu, non_blocking=True)
                images = images.cuda()
                target = target.cuda()

                # compute output
                output = model(images)

                if i == 0:
                    ouput_numpy = output.cpu().numpy()
                    target_numpy = target.cpu().numpy()
                else:
                    ouput_numpy = np.concatenate((ouput_numpy, output.cpu().numpy()), axis=0)
                    target_numpy = np.concatenate((target_numpy, target.cpu().numpy()), axis=0)

                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args_config.print_freq == 0:
                    progress.display(i)

        acc1_all, acc5_all = accuracy_numpy(ouput_numpy, target_numpy, topk=(1, 5))
        print("Epoch: [{}]  Test_{}:  All  Acc@1: {:.2f}  Acc@5: {:.2f}".format(epoch, name, acc1_all, acc5_all))
        if test_os_log:
            print("Epoch: [{}]  Test_{}:  All  Acc@1: {:.2f}  Acc@5: {:.2f}".format(epoch, name, acc1_all, acc5_all), file=test_os_log)

        if name == "iid":
            acc1_all_iid = acc1_all
            acc5_all_iid = acc5_all
        else:
            acc1_all_nui_avg += acc1_all
            acc5_all_nui_avg += acc5_all

    acc1_all_nui_avg /= len(val_loader_nui)
    acc5_all_nui_avg /= len(val_loader_nui)
    print("Epoch: [{}]  All(iid)  Acc@1: {:.2f}  Acc@5: {:.2f}  All(nuisances-avg)  Acc@1: {:.2f}  Acc@5: {:.2f}".format(epoch, acc1_all_iid, acc5_all_iid, acc1_all_nui_avg, acc5_all_nui_avg))
    if test_os_log:
        print("Epoch: [{}]  All(iid)  Acc@1: {:.2f}  Acc@5: {:.2f}  All(nuisances-avg)  Acc@1: {:.2f}  Acc@5: {:.2f}".format(epoch, acc1_all_iid, acc5_all_iid, acc1_all_nui_avg, acc5_all_nui_avg), file=test_os_log)

    return acc1_all_iid, acc5_all_iid, acc1_all_nui_avg, acc5_all_nui_avg


def tester_save_csv(args_main, args_config, class_name, model, val_loader_iid, val_loader_nui):
    file_name = os.path.join(args_config.log_dir, "res")
    file_name_misclassified = os.path.join(args_config.log_dir, "misclassified")
    mkdir_if_missing(file_name)
    mkdir_if_missing(file_name_misclassified)
    val_loader = val_loader_iid + val_loader_nui

    # switch to evaluate mode
    model.eval()
    model = tta.ClassificationTTAWrapper(model, tta.aliases.ten_crop_transform(224, 224))

    for name, loader in val_loader:
        df = pd.read_csv(os.path.join("../starting_ki", name + ".csv"))
        imgs_list = collections.defaultdict(list)

        for id in range(len(df["imgs"])):
            imgs_list[df["imgs"][id]].append(id)

        misclassified_path = []
        with torch.no_grad():
            for i, (images, target, path) in enumerate(loader):
                if args_main.gpu is not None:
                    images = images.cuda(args_main.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args_main.gpu, non_blocking=True)

                # compute output
                output = model(images)
                # print(type(path)) # <class 'list'>
                # print(path) # ['','',...]

                _, pred = output.max(1)
                pred = pred.cpu()

                for j in range(pred.size(0)):
                    label_pred = pred[j].item()
                    label_target = target[j].cpu().item()
                    if label_pred != label_target:
                        misclassified_path.append(path[j])

                for k1 in range(output.size(0)):
                    path_name = path[k1].split("/")[-1]
                    for k2 in imgs_list[path_name]:
                        df["pred"][k2] = class_name[pred[k1].item()]

        with open(os.path.join(file_name_misclassified, name + ".txt"), "w") as f:
            for path in misclassified_path:
                f.write(path + "\n")

        df.to_csv(os.path.join(file_name, name + ".csv"), index=None)


def tester_final_save_csv(args_main, args_config, class_name, model, val_loader):
    file_name = os.path.join(args_config.log_dir, "res_final")
    mkdir_if_missing(file_name)

    df = pd.read_csv(os.path.join("../cls_submission/results.csv"))
    imgs_list = collections.defaultdict(list)

    for id in range(len(df["imgs"])):
        imgs_list[df["imgs"][id]].append(id)

    # switch to evaluate mode
    model.eval()
    model = tta.ClassificationTTAWrapper(model, tta.aliases.ten_crop_transform(224, 224))

    with torch.no_grad():
        for i, (images, path) in enumerate(val_loader):
            if args_main.gpu is not None:
                images = images.cuda(args_main.gpu, non_blocking=True)

            # compute output
            output = model(images)

            _, pred = output.max(1)
            pred = pred.cpu()

            for k1 in range(output.size(0)):
                path_name = path[k1].split("/")[-1]
                for k2 in imgs_list[path_name]:
                    df["pred"][k2] = class_name[pred[k1].item()]

    df.to_csv(os.path.join(file_name, "results.csv"), index=None)











