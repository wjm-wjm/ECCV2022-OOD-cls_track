import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class OnlineLabelSmoothing(nn.Module):
    """
    Implements Online Label Smoothing from paper
    https://arxiv.org/pdf/2011.12562.pdf
    how to use:
    from ols import OnlineLabelSmoothing
    criterion = OnlineLabelSmoothing(alpha=..., n_classes=...)
    for epoch in range(...):  # loop over the dataset multiple times
        for i, data in enumerate(...):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} finished!')
        # Update the soft labels for next epoch
        criterion.next_epoch()
        criterion.eval()
        dev()/test()
    """

    def __init__(self, alpha: float, n_classes: int, smoothing: float = 0.1):
        """
        :param alpha: Term for balancing soft_loss and hard_loss
        :param n_classes: Number of classes of the classification problem
        :param smoothing: Smoothing factor to be used during first epoch in soft_loss
        """
        super(OnlineLabelSmoothing, self).__init__()
        assert 0 <= alpha <= 1, 'Alpha must be in range [0, 1]'
        self.a = alpha
        self.n_classes = n_classes

        # Initialize soft labels with normal LS for first epoch
        self.register_buffer('supervise', torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        # Update matrix is used to supervise next epoch
        self.register_buffer('update', torch.zeros_like(self.supervise))
        # For normalizing we need a count for each class
        self.register_buffer('idx_count', torch.zeros(n_classes))
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: Tensor, y: Tensor):
        # Calculate the final loss
        soft_loss = self.soft_loss(y_h, y)
        hard_loss = self.hard_loss(y_h, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_h: Tensor, y: Tensor):
        """
        Calculates the soft loss and calls step
        to update `update`.
        :param y_h: Predicted logits.
        :param y: Ground truth labels.
        :return: Calculates the soft loss based on current supervise matrix.
        """
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp(), y)
        true_dist = torch.index_select(self.supervise, 1, y).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h, dim=-1))

    def step(self, y_h: Tensor, y: Tensor) -> None:
        """
        Updates `update` with the probabilities
        of the correct predictions and updates `idx_count` counter for
        later normalization.
        Steps:
            1. Calculate correct classified examples.
            2. Filter `y_h` based on the correct classified.
            3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`.
            4. Keep count of # samples added for each `y_h_idx` column.
            5. Average memory by dividing column-wise by result of step (4).
        Note on (5): This is done outside this function since we only need to
                     normalize at the end of the epoch.
        """
        # 1. Calculate predicted classes
        y_h_idx = y_h.argmax(dim=-1)
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, y)
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        # 3. Add y_h probabilities rows as columns to `memory`
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        self.idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def next_epoch(self) -> None:
        """
        This function should be called at the end of the epoch.
        It basically sets the `supervise` matrix to be the `update`
        and re-initializes to zero this last matrix and `idx_count`.
        """
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        self.idx_count[torch.eq(self.idx_count, 0)] = 1  # Avoid 0 denominator
        # Normalize by taking the average
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()


class MaeContrastiveLoss(nn.Module):
    def __init__(self, t=1.0):
        super(MaeContrastiveLoss, self).__init__()
        self.t = t

    def forward(self, input, label):
        """
        x size: [B, L, D]
        label size: [B]
        """
        input_norm = torch.nn.functional.normalize(input, dim=2)
        B, L, D = input_norm.shape
        loss = 0
        for i in range(B):
            anchor = input_norm[i, :1, :]
            positive_sample = input_norm[label == label[i], 1:, :].view(-1, D)
            positive_score = torch.mm(anchor, positive_sample.t())
            negative_sample = input_norm[label != label[i], 1:, :].view(-1, D)
            negative_score = torch.mm(anchor, negative_sample.t())
            loss_temp = - torch.log((positive_score / self.t).exp().sum() / ((positive_score / self.t).exp().sum() + (negative_score / self.t).exp().sum()))
            loss += loss_temp
        loss /= B

        return loss


class MaeConsistentContrastiveLoss(nn.Module):
    def __init__(self, t=1.0):
        super(MaeConsistentContrastiveLoss, self).__init__()
        self.t = t

    def forward(self, input, label):
        """
        x size: [B, L, D]
        label size: [B]
        """
        input_norm = torch.nn.functional.normalize(input, dim=2)
        B, L, D = input_norm.shape
        loss = 0
        criterion_kl = nn.KLDivLoss(reduction="batchmean")
        for i in range(B):
            anchor = input_norm[i, :1, :]
            positive_sample = input_norm[label == label[i], 1:, :].view(-1, D)
            negative_sample = input_norm[label != label[i], 1:, :].view(-1, D)
            anchor_consistent_score = torch.mm(anchor, negative_sample.t())
            positive_consistent_score = torch.mm(positive_sample, negative_sample.t())
            anchor_consistent_score_unify = (anchor_consistent_score / self.t).exp() / (anchor_consistent_score / self.t).exp().sum()
            positive_consistent_score_unify = (positive_consistent_score / self.t).exp() / (anchor_consistent_score / self.t).exp().sum(dim=1, keepdim=True)
            loss_temp1 = criterion_kl(nn.LogSoftmax(1)(anchor_consistent_score_unify.repeat(positive_sample.size(0), 1)), nn.Softmax(1)(positive_consistent_score_unify))
            loss_temp2 = criterion_kl(nn.LogSoftmax(1)(positive_consistent_score_unify), nn.Softmax(1)(anchor_consistent_score_unify.repeat(positive_sample.size(0), 1)))
            loss += (0.5 * loss_temp1 + 0.5 * loss_temp2)
        loss /= B

        return loss


# class CenterLoss(nn.Module):
#     def __init__(self, cls_num, feature_num):
#         super().__init__()
#
#         self.cls_num = cls_num
#         self.feature_num = feature_num
#         self.center = nn.Parameter(torch.rand(cls_num, feature_num)).cuda()
#
#     def forward(self, xs, ys):  # xs=feature, ys=target
#         xs = F.normalize(xs)
#         self.center_exp = self.center.index_select(dim=0, index=ys.long())
#         count = torch.histc(ys, bins=self.cls_num, min=0, max=self.cls_num - 1)
#         self.count_dis = count.index_select(dim=0, index=ys.long()) + 1
#         loss = torch.sum(torch.sum((xs - self.center_exp) ** 2, dim=1) / 2.0 / self.count_dis.float())
#         # print("center:", self.center)
#
#         return loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class CenterTokenLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterTokenLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
