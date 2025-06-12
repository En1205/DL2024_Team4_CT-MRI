import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

def hybrid_loss(pred, target):
    return 0.7 * dice_loss(pred, target) + 0.3 * FocalLoss()(pred, target)

def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred) > threshold
    target = target.bool()
    intersection = (pred & target).float().sum((1, 2, 3))
    union = (pred | target).float().sum((1, 2, 3))
    return ((intersection + eps) / (union + eps)).mean().item()
