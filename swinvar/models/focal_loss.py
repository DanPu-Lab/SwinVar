import torch.nn.functional as F
import torch.nn as nn
import torch

from swinvar.preprocess.parameters import VARIANT


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def smooth(self, inputs, targets):
        num_classes = inputs.size(1)
        with torch.no_grad():
            targets_smooth = torch.full((targets.size(0), num_classes), self.label_smoothing / (num_classes - 1)).to(targets.device).scatter_(1, targets.unsqueeze(1), 1. - self.label_smoothing)

        return targets_smooth

    def forward(self, inputs, targets):

        eps = 1e-8
        # (b, num_classes)
        probs = F.softmax(inputs, dim=-1)
        probs = torch.clamp(probs, min=eps, max=1 - eps)

        # pt = probs.gather(dim=-1, index=targets.view(-1, 1)).view(-1)

        # (b, num_classes)
        ce_loss = -torch.log(probs)

        # (1, num_classes)
        gamma = self.gamma.unsqueeze(0)
        # gamma = 0

        focal_weight = (1 - probs) ** gamma

        # # (b, num_classes)
        alpha_weight = self.alpha.unsqueeze(0)

        targets_smooth = self.smooth(inputs, targets)
        # targets_smooth = 1

        loss = targets_smooth * alpha_weight * focal_weight * ce_loss
        # loss = alpha_weight * ce_loss

        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    def __init__(self, alphas=1.0, gammas=2.0, label_smoothing=0.0, reduction="mean"):
        super(MultiTaskLoss, self).__init__()
        # self.task_weights = nn.Parameter(torch.ones(3))
        self.focal_criterion = [FocalLoss(alphas[i], gammas[i], label_smoothing, reduction) for i in range(len(alphas))]
        # self.ce_criterion = [nn.CrossEntropyLoss() for _ in range(len(alphas))]
        

    def forward(self, predictions, targets):
        
        total_loss = 0.0
        pred_variant_1 = predictions[0]
        pred_variant_2 = predictions[1]
        pred_genotype = predictions[2]
        target_variant_1 = targets[0]
        target_variant_2 = targets[1]
        target_genotype = targets[2]

        loss_genotype = self.focal_criterion[2](pred_genotype, target_genotype)

        loss_variant_1 = 0.0
        loss_variant_2 = 0.0

        task_mask = (target_genotype != 0)

        if task_mask.sum() > 0:
            loss_variant_1 += self.focal_criterion[0](pred_variant_1[task_mask], target_variant_1[task_mask])
            loss_variant_2 += self.focal_criterion[1](pred_variant_2[task_mask], target_variant_2[task_mask])
    
        total_loss = loss_genotype + loss_variant_1 + loss_variant_2
   
        return total_loss


if __name__ == "__main__":
    input_tensor = torch.tensor(range(1, 11), dtype=torch.float32).reshape(2, 5)
    label_tensor = torch.tensor([3, 4], dtype=torch.long)

    probs = F.softmax(input_tensor, dim=-1)
    probs = torch.clamp(probs, min=1e-9, max=1 - 1e-9)
    pt = probs.gather(1, label_tensor.view(-1, 1)).view(-1)
    log_pt = - torch.log(pt)
