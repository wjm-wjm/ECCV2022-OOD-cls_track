import os
import torch
import torchvision.transforms as transforms
from dataset import ROBINDataset_train, ROBINDataset_test, ROBINDataset_test_final, print_dataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image


def build_transform(args_config, is_train):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args_config.input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation='bicubic',
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=mean,
            std=std,
        )

        return transform

    # eval transform
    t = []
    if args_config.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args_config.input_size / crop_pct)
    # size = (256, 256)  # tta
    t.append(
        transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args_config.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)


def data_loading(args_main, args_config):
    train_dir = args_config.train_data
    val_dir_nui = args_config.val_data_nui
    val_dir_iid = args_config.val_data_iid
    val_dir_final = args_config.val_data_final

    normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    # train_transform = transforms.Compose([transforms.RandomResizedCrop(args_config.input_size),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(),
    #                                       normalize,
    #                                       # transforms.RandomErasing(p=0.5),
    #                                       ])
    # test_transform = transforms.Compose([transforms.Resize(256),
    #                                      transforms.CenterCrop(args_config.input_size),
    #                                      transforms.ToTensor(),
    #                                      normalize
    #                                      ])

    train_transform = build_transform(args_config, is_train=True)
    test_transform = build_transform(args_config, is_train=False)

    class2num_list = []

    train_dataset = ROBINDataset_train(os.path.join(train_dir, "Images"), train_transform)
    class2num_list.append(train_dataset.class2num)

    if args_main.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args_config.batch_size, shuffle=(train_sampler is None),
                                                num_workers=args_config.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    # iid validation dataloader
    val_loader_iid = []
    val_dataset_iid = ROBINDataset_test(os.path.join(val_dir_iid, "Images"), test_transform)
    class2num_list.append(val_dataset_iid.class2num)
    val_loader_iid.append(("iid", torch.utils.data.DataLoader(val_dataset_iid,
                           batch_size=args_config.batch_size, shuffle=False,
                           num_workers=args_config.workers, pin_memory=True)))

    # nuisances validation dataloader
    val_nuisances = ["shape", "pose", "texture", "context", "occlusion", "weather"]

    val_loader_nui = []
    for nuisance in val_nuisances:
        val_dataset_nui = ROBINDataset_test(os.path.join(val_dir_nui, nuisance, "Images"), test_transform)
        class2num_list.append(val_dataset_nui.class2num)
        val_loader_nui.append((nuisance, torch.utils.data.DataLoader(val_dataset_nui,
                               batch_size=args_config.batch_size, shuffle=False,
                               num_workers=args_config.workers, pin_memory=True)))

    # final validation dataloader
    val_dataset_final = ROBINDataset_test_final(os.path.join(val_dir_final, "images"), test_transform)
    val_loader_final = torch.utils.data.DataLoader(val_dataset_final,
                               batch_size=args_config.batch_size, shuffle=False,
                               num_workers=args_config.workers, pin_memory=True)

    # print details of dataset
    print_dataset(args_config, class2num_list)

    return train_loader, val_loader_iid, val_loader_nui, val_loader_final, len(train_dataset.classes), train_dataset.classes, train_sampler