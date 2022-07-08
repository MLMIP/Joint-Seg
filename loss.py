import torch.nn as nn

# Dice Loss
class dice_loss(nn.Module):
    def __init__(self, eps=1e-12):
        super(dice_loss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, gt):
        assert pred.size() == gt.size() and pred.size()[1] == 1
        
        N = pred.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
        intersection = pred_flat * gt_flat
        dice = (2.0 * intersection.sum(1) + self.eps) / (pred_flat.sum(1) + gt_flat.sum(1) + self.eps)
        loss = 1.0 - dice.mean()
        
        return loss


# XOR Loss
class xor_loss(nn.Module):
    def __init__(self):
        super(xor_loss, self).__init__()
    
    def forward(self, pred, gt, extra_gt):
        assert pred.size() == gt.size()    and pred.size()[1] == 1
        assert pred.size() == extra_gt.size() and pred.size()[1] == 1
        
        N = pred.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
        extra_gt_flat = extra_gt.view(N, -1)

        overlaps = abs(pred_flat - extra_gt_flat)
        loss = 1 - overlaps.sum(0) / gt_flat.size()
        
        return loss
