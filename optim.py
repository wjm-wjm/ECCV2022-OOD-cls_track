import math
import torch
from torch.optim.lr_scheduler import StepLR


def adjust_learning_rate(args_config, optimizer, epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    epoch = epoch + 1
    if epoch < args_config.warmup_epochs:
        lr = args_config.lr * epoch / args_config.warmup_epochs
    else:
        lr = args_config.min_lr + (args_config.lr - args_config.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args_config.warmup_epochs) / (args_config.epochs - args_config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr


def select_optimizer_lr(args_config, model):
    if args_config.arch == "resnet50":
        ignored_params = list(map(id, model.module.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        # optimizer = torch.optim.SGD([{"params": base_params, "lr": args_config.lr},
        #                              {"params": model.module.classifier.parameters(), "lr": 10 * args_config.lr, "lr_scalse": 10}],
        #                             momentum=args_config.momentum,
        #                             weight_decay=args_config.weight_decay)
        optimizer = torch.optim.AdamW([{"params": base_params, "lr": args_config.lr},
                                       {"params": model.module.classifier.parameters(), "lr": 10 * args_config.lr, "lr_scalse": 10}],
                                      betas=args_config.betas,
                                      weight_decay=args_config.weight_decay)

        # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if args_config.arch == "mae":
        ignored_params = list(map(id, model.module.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        # optimizer = torch.optim.SGD([{"params": base_params, "lr": args_config.lr},
        #                              {"params": model.module.classifier.parameters(), "lr": 10 * args_config.lr, "lr_scalse": 10}],
        #                             momentum=args_config.momentum,
        #                             weight_decay=args_config.weight_decay)
        optimizer = torch.optim.AdamW([{"params": base_params, "lr": args_config.lr},
                                       {"params": model.module.classifier.parameters(), "lr": 10 * args_config.lr, "lr_scalse": 10}],
                                      betas=args_config.betas,
                                      weight_decay=args_config.weight_decay)

        # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if args_config.arch == "resnet50 + mae":
        ignored_params = list(map(id, model.module.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        # optimizer = torch.optim.SGD([{"params": base_params, "lr": args_config.lr},
        #                              {"params": model.module.classifier.parameters(), "lr": 10 * args_config.lr, "lr_scalse": 10}],
        #                             momentum=args_config.momentum,
        #                             weight_decay=args_config.weight_decay)
        optimizer = torch.optim.AdamW([{"params": base_params, "lr": args_config.lr},
                                       {"params": model.module.classifier.parameters(), "lr": 10 * args_config.lr, "lr_scalse": 10}],
                                      betas=args_config.betas,
                                      weight_decay=args_config.weight_decay)

        # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    return optimizer, scheduler