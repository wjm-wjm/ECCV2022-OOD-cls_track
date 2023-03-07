import numpy as np
import torch
import torch.nn as nn
from backbone import resnet, mae
from loss import MaeContrastiveLoss, CenterLoss


def simple_transform(x, beta):
    x = 1/torch.pow(torch.log(1/x+1), beta)

    return x


def extended_simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1), beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1), beta)

    return x_pos + x_neg


class CreateModel(nn.Module):
    def __init__(self, args_config, arch="resnet50", class_num=10):
        super(CreateModel, self).__init__()

        self.args_config = args_config
        self.arch = arch
        self.class_num = class_num

        if self.arch == "resnet50":
            self.pool_dim = 2048
            if args_config.pretrained:
                print("=> using pre-trained model '{}'".format(args_config.arch))

            self.main_net = resnet.resnet50(pretrained=args_config.pretrained)

            self.layer0 = nn.Sequential(self.main_net.conv1, self.main_net.bn1, self.main_net.relu, self.main_net.maxpool)
            self.layer1 = self.main_net.layer1
            self.layer2 = self.main_net.layer2
            self.layer3 = self.main_net.layer3
            self.layer4 = self.main_net.layer4

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            self.classifier = nn.Linear(self.pool_dim, self.class_num)

        if self.arch == "mae":
            # self.pool_dim = 768
            # self.main_net = mae.mae_vit_base_patch16(norm_pix_loss=False)
            self.pool_dim = 1024
            self.main_net = mae.mae_vit_large_patch16(norm_pix_loss=False)
            # self.pool_dim = 1280
            # self.main_net = mae.mae_vit_huge_patch14(norm_pix_loss=False)
            if args_config.pretrained:
                print("=> using pre-trained model '{}'".format(args_config.arch))
                checkpoint = torch.load(args_config.pretrained_model_path)
                self.main_net.load_state_dict(checkpoint["model"])
                # model_dict = self.main_net.state_dict()
                # pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k in model_dict}
                # model_dict.update(pretrained_dict)
                # self.main_net.load_state_dict(model_dict)

            self.classifier = nn.Linear(self.pool_dim, self.class_num)

        if self.arch == "resnet50 + mae":
            self.pool_dim = 768 + 2048
            self.main_resnet50 = resnet.resnet50(pretrained=args_config.pretrained)
            self.main_mae = mae.mae_vit_base_patch16(norm_pix_loss=False)
            if args_config.pretrained:
                print("=> using pre-trained model '{}'".format(args_config.arch))
                checkpoint = torch.load(args_config.pretrained_mae_path)
                self.main_mae.load_state_dict(checkpoint["model"])

            self.layer0 = nn.Sequential(self.main_resnet50.conv1, self.main_resnet50.bn1, self.main_resnet50.relu, self.main_resnet50.maxpool)
            self.layer1 = self.main_resnet50.layer1
            self.layer2 = self.main_resnet50.layer2
            self.layer3 = self.main_resnet50.layer3
            self.layer4 = self.main_resnet50.layer4

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            self.classifier = nn.Linear(self.pool_dim, self.class_num)

    def forward(self, x, target=None, target_rand=None, lam=None):
        if self.arch == "resnet50":
            x_0 = self.layer0(x)
            x_1 = self.layer1(x_0)
            x_2 = self.layer2(x_1)
            x_3 = self.layer3(x_2)
            x_4 = self.layer4(x_3)

            x_pool_4 = self.avgpool(x_4)
            x_pool_4 = x_pool_4.view(x_pool_4.size(0), x_pool_4.size(1))

            p_4 = self.classifier(x_pool_4)

            if self.training:
                ce_loss = self.cal_loss(p_4, target, latent=None, target_rand=target_rand, lam=lam)

                return p_4, ce_loss
            else:
                return p_4

        if self.arch == "mae":
            if self.training:
                mae_loss, latent, pred, imgs, imgs_mask = self.main_net(x, mask_ratio=0.5, iscutmix=self.args_config.cutmix)
                p = self.classifier(latent[:, 0, :])
                ce_loss, mnce_loss = self.cal_loss(p, target, latent=latent, target_rand=target_rand, lam=lam)

                return p, mae_loss, ce_loss, mnce_loss, pred, imgs, imgs_mask
            else:
                latent = self.main_net.forward_encoder_test(x)
                p = self.classifier(latent[:, 0, :])

                return p

        if self.arch == "resnet50 + mae":
            if self.training:
                mae_loss, latent, pred, imgs, imgs_mask = self.main_mae(x, mask_ratio=0.5)
                x_0 = self.layer0(pred)
                x_1 = self.layer1(x_0)
                x_2 = self.layer2(x_1)
                x_3 = self.layer3(x_2)
                x_4 = self.layer4(x_3)

                x_pool_4 = self.avgpool(x_4)
                x_pool_4 = x_pool_4.view(x_pool_4.size(0), x_pool_4.size(1))

                p = self.classifier(torch.cat((latent[:, 0, :], x_pool_4), dim=1))

                ce_loss, mnce_loss = self.cal_loss(p, target, latent=latent, target_rand=target_rand, lam=lam)

                return p, mae_loss, ce_loss, mnce_loss, pred, imgs, imgs_mask
            else:
                latent = self.main_mae.forward_encoder_test(x)
                x_0 = self.layer0(x)
                x_1 = self.layer1(x_0)
                x_2 = self.layer2(x_1)
                x_3 = self.layer3(x_2)
                x_4 = self.layer4(x_3)

                x_pool_4 = self.avgpool(x_4)
                x_pool_4 = x_pool_4.view(x_pool_4.size(0), x_pool_4.size(1))

                p = self.classifier(torch.cat((latent[:, 0, :], x_pool_4), dim=1))

                return p

    def cal_loss(self, output, target, latent=None, target_rand=None, lam=None):
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.0)
        criterion_mnce = MaeContrastiveLoss(t=1.0)
        criterion_kl = nn.KLDivLoss(reduction="batchmean")

        if self.args_config.cutmix:
            ce_loss = criterion_ce(output, target) * lam + criterion_ce(output, target_rand) * (1. - lam)
            # ce_loss = criterion_ce(output, target)
        else:
            ce_loss = criterion_ce(output, target)

        if self.arch == "resnet50":

            return ce_loss

        if self.arch == "mae":
            if self.args_config.cutmix:
                mnce_loss = criterion_mnce(latent, target) * lam + criterion_mnce(latent, target_rand) * (1. - lam)
                # mnce_loss = criterion_mnce(latent, target)
            else:
                mnce_loss = criterion_mnce(latent, target)

            return ce_loss, mnce_loss

        if self.arch == "resnet50 + mae":
            if self.args_config.cutmix:
                mnce_loss = criterion_mnce(latent, target) * lam + criterion_mnce(latent, target_rand) * (1. - lam)
                # mnce_loss = criterion_mnce(latent, target)
            else:
                mnce_loss = criterion_mnce(latent, target)

            return ce_loss, mnce_loss




