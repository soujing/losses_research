import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt
import cv2 as cv
import numpy as np
import os
os.chdir("/home/jsj0414/")

DEVICE = torch.device("cuda") # select device for training, i.e. gpu or cpu

##################################################################
#######################  single - loss  ########################
##################################################################
        
class IoULoss(nn.Module): # 남겨
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1) # &&

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)

        return 1 -IoU
        
class DiceLoss(nn.Module): # 남겨
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # dict형태로 데이터가 들어오는 경우가 있음 #
        # if isinstance(inputs, dict):
        #     inputs = torch.sigmoid(inputs['out'])
        # elif isinstance(inputs, tuple):
        #     inputs = torch.sigmoid(inputs[0])
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)  
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        dice = (2.*intersection + smooth)/(total + smooth)  
        return 1 - dice

class BCELoss(nn.Module): # 남겨
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE
        
class FocalLoss(nn.Module): # 남겨
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs) 
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE        
        
        return focal_loss


##################################################################
#######################  2 - combination  ########################
##################################################################

class IoUDiceLoss(nn.Module): # 남겨
    def __init__(self):
        super(IoUDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        dice = (2.*intersection + smooth)/(total + smooth)                  

        return (1-IoU) + (1-dice)

class IoUBCELoss(nn.Module): # 남겨
    def __init__(self):
        super(IoUBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return (1-IoU) + BCE

class IoUFocalLoss(nn.Module): # 남겨
    def __init__(self):
        super(IoUFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE        

        return (1-IoU) + focal_loss

class DiceBCELoss(nn.Module): # 남겨
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return (1-dice) + BCE

class DiceFocalLoss(nn.Module): # 남겨
    def __init__(self):
        super(DiceFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE        

        return (1-dice) + focal_loss

class BCEFocalLoss(nn.Module): # 남겨
    def __init__(self):
        super(BCEFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE        

        return BCE + focal_loss

##################################################################
#######################  3 - combination  ########################
##################################################################

class IoUBCEFocalLoss(nn.Module): # 남겨
    def __init__(self):
        super(IoUBCEFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE        

        return (1-IoU) + BCE + focal_loss

class IoUFocalL1Loss(nn.Module): # 남겨
    def __init__(self):
        super(IoUFocalL1Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l1=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        # L1 term
        L1_reg = torch.abs(inputs - targets).sum()

        return (1-IoU) + focal_loss + lambda_l1 * L1_reg

class IoUFocalL2Loss(nn.Module): # 남겨
    def __init__(self):
        super(IoUFocalL2Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l2=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE        

        # L2 term (squared difference)
        L2_reg = torch.pow(inputs - targets, 2).sum()

        return (1-IoU) + focal_loss + lambda_l2 * L2_reg

class DiceBCEFocalLoss(nn.Module): # 남겨
    def __init__(self):
        super(DiceBCEFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE        

        return (1-dice) + BCE + focal_loss

class DiceBCEL1Loss(nn.Module): # 남겨
    def __init__(self):
        super(DiceBCEL1Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6, lambda_l1=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # L1 term
        L1_reg = torch.abs(inputs - targets).sum()

        return (1-dice) + BCE + lambda_l1 * L1_reg

class DiceBCEL2Loss(nn.Module): # 남겨
    def __init__(self):
        super(DiceBCEL2Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6, lambda_l2=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # L2 term (squared difference)
        L2_reg = torch.pow(inputs - targets, 2).sum()

        return (1-dice) + BCE + lambda_l2 * L2_reg


##################################################################
#######################  4 - combination  ########################
##################################################################

class IoUBCEFocalL1Loss(nn.Module): # 남겨
    def __init__(self):
        super(IoUBCEFocalL1Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l1=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        # L1 term
        L1_reg = torch.abs(inputs - targets).sum()

        return (1-IoU) + BCE + focal_loss + lambda_l1 * L1_reg

class DiceBCEFocalL1Loss(nn.Module): # 남겨
    def __init__(self):
        super(DiceBCEFocalL1Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l1=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        # L1 term
        L1_reg = torch.abs(inputs - targets).sum()        

        return (1-dice) + BCE + focal_loss + lambda_l1 * L1_reg


class IoUBCEFocalL2Loss(nn.Module): # 남겨
    def __init__(self):
        super(IoUBCEFocalL2Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l2=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        # L2 term (squared difference)
        L2_reg = torch.pow(inputs - targets, 2).sum()

        return (1-IoU) + BCE + focal_loss + lambda_l2 * L2_reg

class IoUFocalL1L2Loss(nn.Module): # 남겨
    def __init__(self):
        super(IoUFocalL1L2Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l1=0.01, lambda_l2=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        # L1 term
        L1_reg = torch.abs(inputs - targets).sum()

        # L2 term (squared difference)
        L2_reg = torch.pow(inputs - targets, 2).sum()        

        return (1-IoU) + focal_loss + lambda_l1 * L1_reg + lambda_l2 * L2_reg

class DiceBCEFocalL2Loss(nn.Module): # 남겨
    def __init__(self):
        super(DiceBCEFocalL2Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l2=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        # L2 term (squared difference)
        L2_reg = torch.pow(inputs - targets, 2).sum()   

        return (1-dice) + BCE + focal_loss + lambda_l2 * L2_reg

class DiceBCEL1L2Loss(nn.Module): # 남겨
    def __init__(self):
        super(DiceBCEL1L2Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6, lambda_l1=0.01, lambda_l2=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # L1 term
        L1_reg = torch.abs(inputs - targets).sum()

        # L2 term (squared difference)
        L2_reg = torch.pow(inputs - targets, 2).sum()

        return (1-dice) + BCE + lambda_l1 * L1_reg + lambda_l2 * L2_reg

##################################################################
#######################  5 - combination  ########################
##################################################################

class IoUBCEFocalL1L2Loss(nn.Module): # 남겨
    def __init__(self):
        super(IoUBCEFocalL1Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l1=0.01, lambda_l2=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        # L1 term
        L1_reg = torch.abs(inputs - targets).sum()

        # L2 term (squared difference)
        L2_reg = torch.pow(inputs - targets, 2).sum()        

        return (1-IoU) + BCE + focal_loss + lambda_l1 * L1_reg + lambda_l2 * L2_reg

class DiceBCEFocalL1L2Loss(nn.Module): # 남겨
    def __init__(self):
        super(DiceBCEFocalL1Loss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6, lambda_l1=0.01, lambda_l2=0.01):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice = (2.*intersection + smooth)/(total + smooth)          
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        # L1 term
        L1_reg = torch.abs(inputs - targets).sum()

        # L2 term (squared difference)
        L2_reg = torch.pow(inputs - targets, 2).sum()

        return (1-dice) + BCE + focal_loss + lambda_l1 * L1_reg + lambda_l2 * L2_reg


##################################################################
#######################  T-Loss  ########################
##################################################################

class TLoss(nn.Module):
    def __init__(
        self,
        image_size = 224,
        device = DEVICE,
        nu: float = 1.0,
        epsilon: float = 1e-8,
        reduction: str = "mean",
    ):
        """
        Implementation of the TLoss.

        Args:
            config: Configuration object for the loss.
            image_size: width (= height)
            device
            nu (float): Value of nu.
            epsilon (float): Value of epsilon.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super().__init__()
        self.D = torch.tensor(
            (image_size * image_size),
            dtype=torch.float,
            device=device
        )
 
        self.lambdas = torch.ones(
            (image_size, image_size),
            dtype=torch.float,
            device=device
        )
        self.nu = nn.Parameter(
            torch.tensor(nu, dtype=torch.float, device=device)
        )
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=device)
        self.reduction = reduction

    def forward(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Model's prediction, size (B x W x H).
            target_tensor (torch.Tensor): Ground truth, size (B x W x H).

        Returns:
            torch.Tensor: Total loss value.
        """
        # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
        if isinstance(input_tensor, dict):
            input_tensor = torch.sigmoid(input_tensor['out'])

        elif isinstance(input_tensor, tuple):
            input_tensor = torch.sigmoid(input_tensor[0])
        
        else:
            input_tensor = torch.sigmoid(input_tensor)               

        delta_i = input_tensor - target_tensor
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = (
            first_term
            + second_term
            + third_term
            + fourth_term
            + fifth_term
            + sixth_term
        )

        if self.reduction == "mean":
            total_loss = total_losses.mean()
        elif self.reduction == "sum":
            total_loss = total_losses.sum()
        elif self.reduction == "none":
            total_loss = total_losses
        else:
            raise ValueError(
                f"The reduction method '{self.reduction}' is not implemented."
            )

        smooth=1
        lambda_reg=0.01

        # 레이블과 예측 텐서를 평탄화
        input_tensor = input_tensor.view(-1)
        target_tensor = target_tensor.view(-1)
        
        intersection = (input_tensor * target_tensor).sum()                            
        dice = (2. * intersection + smooth) / (input_tensor.sum() + target_tensor.sum() + smooth)
        BCE = F.binary_cross_entropy(input_tensor, target_tensor, reduction='mean')
        LogDiceBCE = BCE - 1.0 * torch.log(dice)
        
        # L1 정규화 항 계산
        L1_reg = torch.abs(input_tensor - target_tensor).sum()
        
        # 최종 손실 함수에 L1 정규화 항 추가
        total_loss = total_loss/10000 + LogDiceBCE + lambda_reg * L1_reg   
        total_loss = total_loss / (total_loss.max() + 1e-6)
        return total_loss




##################################################################
#######################  지역정보를 반영  ########################
##################################################################
class IoUWithTVLoss(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(IoUWithTVLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # Compute IoU Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        total = (inputs + targets).sum(dim=(1, 2, 3))
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou.mean()

        # Compute Total Variation Loss
        dx = torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        total_loss = iou_loss + self.tv_weight * tv_loss
        return total_loss

class IoUWithTV_tar_Loss(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(IoUWithTV_tar_Loss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # Compute IoU Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        total = (inputs + targets).sum(dim=(1, 2, 3))
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou.mean()

        # Compute Total Variation Loss
        dx = torch.abs(inputs[:, :, 1:, :] - targets[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - targets[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        total_loss = iou_loss + self.tv_weight * tv_loss
        return total_loss
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithTVLoss(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(BCEWithTVLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, smooth=1e-6):
        # Apply Sigmoid if needed (comment out if your model already applies Sigmoid)
        inputs = torch.sigmoid(inputs)

        # Compute Binary Cross-Entropy (BCE) Loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # Compute Total Variation (TV) Loss
        dx = torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        # Total Loss
        total_loss = BCE + self.tv_weight * tv_loss
        return total_loss

class BCEWithTVLoss_tar(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(BCEWithTVLoss_tar, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, smooth=1e-6):
        # Apply Sigmoid if needed (comment out if your model already applies Sigmoid)
        inputs = torch.sigmoid(inputs)

        # Compute Binary Cross-Entropy (BCE) Loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # Compute Total Variation (TV) Loss
        dx = torch.abs(inputs[:, :, 1:, :] - targets[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - targets[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        # Total Loss
        total_loss = BCE + self.tv_weight * tv_loss
        return total_loss
        

sys.path.append("./.conda/envs/image/lib/python3.12/site-packages/")
import pytorch_ssim

class BCEWithSSIMLoss(nn.Module):
    def __init__(self, ssim_weight=0.1):
        super(BCEWithSSIMLoss, self).__init__()
        self.ssim_weight = ssim_weight

    def forward(self, inputs, targets, smooth=1e-6):
        # Apply Sigmoid if needed
        inputs = torch.sigmoid(inputs)

        # Compute Binary Cross-Entropy (BCE) Loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # Compute SSIM Loss (inputs and targets must be 4D tensors)
        ssim_loss = 1 - pytorch_ssim.ssim(inputs, targets)

        # Combine BCE and SSIM Loss
        total_loss = bce_loss + self.ssim_weight * ssim_loss
        return total_loss
        

sys.path.append("./.conda/envs/image/lib/python3.12/site-packages/")
import pytorch_ssim

class IoUWithSSIMLoss(nn.Module):
    def __init__(self, ssim_weight=0.1):
        super(IoUWithSSIMLoss, self).__init__()
        self.ssim_weight = ssim_weight

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # Compute IoU Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        total = (inputs + targets).sum(dim=(1, 2, 3))
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou.mean()

        # Compute SSIM Loss
        ssim_loss = 1 - pytorch_ssim.ssim(inputs, targets)

        total_loss = iou_loss + self.ssim_weight * ssim_loss
        return total_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

class BoundaryUncertaintyLoss(nn.Module):
    def __init__(self, boundary_width=3):
        """
        Args:
            boundary_width: 경계 마스크의 너비.
        """
        super(BoundaryUncertaintyLoss, self).__init__()
        self.boundary_width = boundary_width
        
    def forward(self, predictions, labels):
        """
        Args:
            predictions: [B, C, H, W] - 모델의 예측 값 (logits).
            labels: [B, H, W] - Ground truth 마스크 (binary or multi-class).

        Returns:
            Boundary Uncertainty Loss 값.
        """
        # 소프트맥스를 적용해 확률 계산
        probs = F.softmax(predictions, dim=1)  # [B, C, H, W]

        # 불확실성 계산 (Entropy)
        uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B, H, W]

        # Boundary mask 생성
        boundary_mask = self._generate_boundary_mask(labels)

        # Boundary 영역의 불확실성 손실 계산
        boundary_uncertainty = uncertainty * boundary_mask
        loss = boundary_uncertainty.mean()

        return loss

    def _generate_boundary_mask(self, labels):
        """
        Boundary mask 생성 함수. 거리 변환을 이용하여 경계 영역 추출.
        Args:
            labels: [B, H, W] - Ground truth 마스크 (binary or multi-class).

        Returns:
            Boundary mask: [B, H, W] - 경계 영역 마스크 (0 또는 1).
        """
        boundary_masks = []
        for label in labels.cpu().numpy():
            # 각 클래스의 경계를 추출
            boundary = (label > 0).astype(float) - (distance_transform_edt(label == 0) <= self.boundary_width).astype(float)
            boundary_mask = (boundary != 0).astype(float)
            boundary_masks.append(boundary_mask)
        
        boundary_masks = torch.tensor(boundary_masks, dtype=torch.float32, device=labels.device)
        return boundary_masks

# Spatially Weighted Loss
import torch
import torch.nn as nn

class SpatiallyWeightedLoss(nn.Module):
    def __init__(self):
        """
        Args:
            base_loss_fn: PyTorch 기본 손실 함수 (예: nn.CrossEntropyLoss(reduction='none')).
        """
        super(SpatiallyWeightedLoss, self).__init__()
        self.base_loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets, weights=0.5):
        """
        Args:
            predictions: [B, C, H, W] - 모델의 예측 값 (logits).
            targets: [B, H, W] - Ground truth 라벨.
            weights: [B, H, W] - 각 픽셀별 가중치 (기본값: 1).

        Returns:
            Spatially weighted loss 값 (scalar).
        """
        # # 기본 가중치 설정 (가중치가 없는 경우 모두 1로 설정)
        # if weights is None:
        #     weights = torch.ones_like(targets, dtype=torch.float32, device=targets.device)  # [B, H, W]

        # 기본 손실 계산
        base_loss = self.base_loss_fn(predictions, targets)  # [B, H, W]

        # 가중치 적용
        weighted_loss = base_loss * weights  # [B, H, W]

        # 손실의 평균값 반환
        return weighted_loss.mean()


# Saliency-guided Loss
class SaliencyGuidedLoss(nn.Module):
    def __init__(self, base_loss_fn=None):
        super(SaliencyGuidedLoss, self).__init__()
        # 기본 loss를 CrossEntropyLoss로 설정하거나 외부에서 받음
        self.base_loss_fn = base_loss_fn or nn.CrossEntropyLoss(reduction='none')

    
    def forward(self, predictions, images, targets):
        """
        Args:
            predictions: [B, C, H, W] - 모델의 예측 값.
            images: [B, C, H, W] - 입력 이미지.
            targets: [B, H, W] - Ground truth 라벨.

        Returns:
            Saliency-guided loss 값.
        """
        # Saliency map 생성 (사용자가 구현 필요)
        saliency_map = generate_saliency_map(images)  # [B, H, W]
        
        # Base loss 계산
        base_loss = self.base_loss_fn(predictions, targets)  # [B, H, W]

        # 크기 일치 확인
        if saliency_map.shape != base_loss.shape:
            raise ValueError(f"Saliency map shape {saliency_map.shape}와 base_loss shape {base_loss.shape}가 일치하지 않습니다.")

        # Weighted loss 계산
        weighted_loss = base_loss * saliency_map  # [B, H, W]
        return weighted_loss.mean()


# Graph-based Loss
class GraphBasedLoss(nn.Module):
    def __init__(self):
        super(GraphBasedLoss, self).__init__()

    def forward(self, predictions, adjacency_matrix):
        """
        Args:
            predictions: [N, C] - 노드별 예측 값 (Flattened features).
            adjacency_matrix: [N, N] - 그래프의 인접 행렬.

        Returns:
            Graph-based loss 값.
        """
        # 노드 간의 차이 계산
        diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [N, N, C]
        weighted_diff = adjacency_matrix.unsqueeze(-1) * diff.pow(2)  # 가중치 적용
        return weighted_diff.mean()


# IoUWithEdgeLoss
class IoUWithEdgeLoss(nn.Module):
    def __init__(self, edge_weight=0.1, device=DEVICE):
        super(IoUWithEdgeLoss, self).__init__()
        self.edge_weight = edge_weight
        self.device = device
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(self.device)  # 필터를 초기화와 동시에 device로 이동
        sobel_kernel = [[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]]
        self.edge_filter.weight = nn.Parameter(
            torch.tensor(sobel_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # 커널도 device로 이동
        )

    def forward(self, inputs, targets, smooth=1e-6):
        # 모든 텐서를 동일한 디바이스로 이동
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Sigmoid activation
        inputs = torch.sigmoid(inputs)

        # Compute IoU Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        total = (inputs + targets).sum(dim=(1, 2, 3))
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou.mean()

        # Compute Edge Loss
        inputs_edge = torch.abs(self.edge_filter(inputs))
        targets_edge = torch.abs(self.edge_filter(targets))
        edge_loss = nn.functional.l1_loss(inputs_edge, targets_edge)

        # Combine IoU and Edge Loss
        total_loss = iou_loss + self.edge_weight * edge_loss
        return total_loss




## IoUWithContextualLoss
from torchvision import models

class IoUWithContextualLoss(nn.Module):
    def __init__(self, cx_weight=0.1, device=DEVICE):
        super(IoUWithContextualLoss, self).__init__()
        self.cx_weight = cx_weight
        self.device = device

        # Load VGG16 model and use features up to layer 16
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval().to(self.device)

        # Freeze parameters in the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets, smooth=1e-6):
        # Ensure inputs and targets are on the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Convert single-channel inputs/targets to 3-channel for VGG
        if inputs.shape[1] == 1:  # Check if the channel dimension is 1
            inputs = inputs.repeat(1, 3, 1, 1)  # Duplicate across 3 channels
            targets = targets.repeat(1, 3, 1, 1)

        # Apply sigmoid activation to inputs
        inputs = torch.sigmoid(inputs)

        # Compute IoU Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        total = (inputs + targets).sum(dim=(1, 2, 3))
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou.mean()

        # Compute Contextual Loss
        with torch.no_grad():  # Disable gradient computation for feature extraction
            inputs_features = self.feature_extractor(inputs)
            targets_features = self.feature_extractor(targets)
        contextual_loss = nn.functional.mse_loss(inputs_features, targets_features)

        # Combine IoU Loss and Contextual Loss
        total_loss = iou_loss + self.cx_weight * contextual_loss
        return total_loss




## WeightedIoULoss
class WeightedIoULoss(nn.Module):
    def __init__(self, weight_map=None):
        super(WeightedIoULoss, self).__init__()
        self.weight_map = weight_map

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        if self.weight_map is not None:
            weights = self.weight_map
        else:
            # Compute weights based on gradient magnitude
            dx = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
            dy = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])

            # Align sizes of dx and dy
            dx = dx[:, :, :, :-1]  # Match width with dy
            dy = dy[:, :, :-1, :]  # Match height with dx

            gradient_magnitude = dx + dy
            weights = gradient_magnitude.detach()

            # Pad weights to match target size
            weights = torch.nn.functional.pad(weights, (0, 1, 0, 1))

        # Compute weighted IoU Loss
        intersection = ((inputs * targets) * weights).sum(dim=(1, 2, 3))
        total = ((inputs + targets) * weights).sum(dim=(1, 2, 3))
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou.mean()

        return iou_loss


# DiceWithTVLoss inputs-inputs
class DiceWithTVLoss_in(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(DiceWithTVLoss_in, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # Compute Dice Loss without flattening
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        # Compute Total Variation Loss
        dx = torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        total_loss = dice_loss + self.tv_weight * tv_loss
        return total_loss

# DiceWithTVLoss inputs-targets
class DiceWithTVLoss_tar(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(DiceWithTVLoss_tar, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # Compute Dice Loss without flattening
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        # Compute Total Variation Loss
        dx = torch.abs(inputs[:, :, 1:, :] - targets[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - targets[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        total_loss = dice_loss + self.tv_weight * tv_loss
        return total_loss


# DiceWithSSIMLoss
import pytorch_ssim

class DiceWithSSIMLoss(nn.Module):
    def __init__(self, ssim_weight=0.1):
        super(DiceWithSSIMLoss, self).__init__()
        self.ssim_weight = ssim_weight

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # Compute Dice Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        # Compute SSIM Loss
        ssim_loss = 1 - pytorch_ssim.ssim(inputs, targets)

        total_loss = dice_loss + self.ssim_weight * ssim_loss
        return total_loss


# DiceWithEdgeLoss
class DiceWithEdgeLoss(nn.Module):
    def __init__(self, edge_weight=0.1, device=DEVICE):
        super(DiceWithEdgeLoss, self).__init__()
        self.edge_weight = edge_weight
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.device = device
        sobel_kernel = [[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]]
        self.edge_filter.weight = nn.Parameter(torch.tensor(sobel_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device))

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # Compute Dice Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        # Compute Edge Loss
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs_edge = torch.abs(self.edge_filter(inputs))
        targets_edge = torch.abs(self.edge_filter(targets))
        edge_loss = nn.functional.l1_loss(inputs_edge, targets_edge)

        total_loss = dice_loss + self.edge_weight * edge_loss
        return total_loss

# DiceWithContextualLoss
from torchvision import models

class DiceWithContextualLoss(nn.Module):
    def __init__(self, cx_weight=0.1,device = DEVICE):
        super(DiceWithContextualLoss, self).__init__()
        self.cx_weight = cx_weight
        self.device =device
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval().to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs = torch.sigmoid(inputs).to(self.device)

        # Compute Dice Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        # Compute Contextual Loss
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs_features = self.feature_extractor(inputs.repeat(1, 3, 1, 1))
        targets_features = self.feature_extractor(targets.repeat(1, 3, 1, 1))
        contextual_loss = nn.functional.mse_loss(inputs_features, targets_features)

        total_loss = dice_loss + self.cx_weight * contextual_loss
        return total_loss

# MultiScaleDiceLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDiceLoss(nn.Module):
    def __init__(self, weights=[1.0, 0.5, 0.25], smooth=1e-6):
        super(MultiScaleDiceLoss, self).__init__()
        self.weights = weights
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid activation if not already applied
        inputs = torch.sigmoid(inputs)

        total_loss = 0.0

        # Compute Dice loss at the original scale
        x0 = self.dice_loss(inputs, targets)
        total_loss += self.weights[0] * x0

        # Downsample inputs and targets to half size
        inputs_down1 = F.interpolate(inputs, scale_factor=0.5, mode='bilinear', align_corners=False)
        targets_down1 = F.interpolate(targets, scale_factor=0.5, mode='nearest')
        x1 = self.dice_loss(inputs_down1, targets_down1)
        total_loss += self.weights[1] * x1

        # Downsample inputs and targets to quarter size
        inputs_down2 = F.interpolate(inputs, scale_factor=0.25, mode='bilinear', align_corners=False)
        targets_down2 = F.interpolate(targets, scale_factor=0.25, mode='nearest')
        x2 = self.dice_loss(inputs_down2, targets_down2)
        total_loss += self.weights[2] * x2

        return total_loss

    def dice_loss(self, inputs, targets):
        smooth = self.smooth

        # Flatten the tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # Compute the Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        dice = (2. * intersection + smooth) / (union + smooth)

        return 1 - dice

# FocalWithTVLoss_in
class FocalWithTVLoss_in(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(FocalWithTVLoss_in, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        # Sigmoid 적용 (모델 출력이 확률이 아닌 경우)
        inputs = torch.sigmoid(inputs) 
        
        # Focal Loss 계산
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE        

        # Total Variation Loss 계산
        dx = torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        # Focal Loss + TV Loss
        total_loss = focal_loss + self.tv_weight * tv_loss
        return total_loss

# FocalWithTVLoss
class FocalWithTVLoss_tar(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(FocalWithTVLoss_tar, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        # Sigmoid 적용 (모델 출력이 확률이 아닌 경우)
        inputs = torch.sigmoid(inputs) 
        
        # Focal Loss 계산
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE        

        # Compute Total Variation (TV) Loss
        dx = torch.abs(inputs[:, :, 1:, :] - targets[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - targets[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        # Focal Loss + TV Loss
        total_loss = focal_loss + self.tv_weight * tv_loss
        return total_loss
        
# FocalWithGaussianLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalWithGaussianLoss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, weight=0.1, device=DEVICE):
        super(FocalWithGaussianLoss, self).__init__()
        self.kernel_size = kernel_size
        self.gaussian_kernel = self._create_gaussian_kernel(kernel_size, sigma).to(device)
        self.weight = weight
        self.device = device

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # Create Gaussian kernel
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss = gauss / gauss.sum()
        kernel = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)
        return kernel

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        # Apply Sigmoid if needed
        inputs = torch.sigmoid(inputs).to(self.device)
        targets = targets.to(self.device)

        # Flatten inputs for Focal Loss
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        # Ensure inputs are 4D for Conv2d
        if inputs.dim() < 4:
            inputs = inputs.unsqueeze(0).unsqueeze(0)

        # Apply Gaussian smoothing
        padding = self.kernel_size // 2
        smoothed_inputs = F.conv2d(inputs, self.gaussian_kernel, padding=padding)
        smoothing_loss = torch.mean((inputs - smoothed_inputs) ** 2)

        # Combine Focal Loss and Smoothing Loss
        total_loss = focal_loss + self.weight * smoothing_loss
        return total_loss


# FocalWithRegionWeighting
class FocalWithRegionWeighting(nn.Module):
    def __init__(self, region_weight=2.0):
        super(FocalWithRegionWeighting, self).__init__()
        self.region_weight = region_weight

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6): # region_mask
        
        # region_mask는 배경은 0, 강조할 부분은 1이 되기 때문
        region_mask = (inputs >= 0.5).float()
        
        # Sigmoid 적용
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        region_mask_flat = region_mask.view(-1)

        # Weighted Focal Loss
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        # Apply region weights
        weighted_loss = focal_loss * (1 + self.region_weight * region_mask_flat)
        return weighted_loss.mean()



# # Topology-aware Loss
# class TopologyAwareLoss(nn.Module):
#     def __init__(self):
#         super(TopologyAwareLoss, self).__init__()

#     def forward(self, predictions, labels):
#         """
#         Args:
#             predictions: [B, C, H, W] - 모델의 예측 값.
#             topology_map: [B, H, W] - 위상 정보를 포함한 맵.

#         Returns:
#             Topology-aware loss 값.
#         """
#         # MSE 손실을 사용하여 위상 정보 보존
#         topology_map = labels.reshape((B, H, W)) 
#         loss = F.mse_loss(predictions, topology_map)
#         return loss


# crf손실함수
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("./.conda/envs/image/lib/python3.12/site-packages")
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

class CRFLoss(nn.Module):
    def __init__(self, spatial_sigma=3, color_sigma=13, num_iterations=5):
        super(CRFLoss, self).__init__()
        self.spatial_sigma = spatial_sigma  # Pairwise Gaussian standard deviation
        self.color_sigma = color_sigma  # Pairwise Bilateral standard deviation
        self.num_iterations = num_iterations

    def forward(self, logits, images, labels):
        """
        Args:
            logits: [B, C, H, W] - Predicted logits from the model.
            images: [B, 3, H, W] - Original input images (used for bilateral term).
            labels: [B, H, W] - Ground truth labels.

        Returns:
            CRF loss value.
        """
        b, c, h, w = logits.size()
        loss = 0.0

        # Loop over batch
        for i in range(b):
            # Convert PyTorch tensors to numpy arrays
            unary = unary_from_softmax(logits[i].softmax(dim=0).detach().cpu().numpy())
            img = images[i].permute(1, 2, 0).cpu().numpy()  # Convert to HxWx3
            img = np.ascontiguousarray(img) 
            img = (img * 255).clip(0, 255).astype(np.uint8)  # Ensure img is uint8

            label = labels[i].cpu().numpy()  # Ground truth in NumPy format
    
            # Initialize DenseCRF
            d = dcrf.DenseCRF2D(w, h, c)
            d.setUnaryEnergy(unary)
    
            # Add pairwise potentials (spatial and bilateral)
            d.addPairwiseGaussian(sxy=self.spatial_sigma, compat=3)
            d.addPairwiseBilateral(sxy=self.color_sigma, srgb=self.color_sigma, rgbim=img, compat=10)
    
            # Perform CRF inference
            q = d.inference(self.num_iterations)
            q = np.array(q).reshape((c, h, w))  # Reshape to [C, H, W]
    
            # Convert Q (CRF output) and labels back to PyTorch tensors
            crf_output = torch.tensor(q, dtype=torch.float32, device=logits.device)

            # Convert probabilities to logits
            crf_output_logits = torch.log(crf_output + 1e-8)

            # Convert labels to tensor
            label_tensor = torch.tensor(label, dtype=torch.long, device=logits.device)
            label_tensor = label_tensor.to(torch.long)
    
            # Compute pixel-wise cross-entropy loss
            crf_loss = nn.CrossEntropyLoss()(crf_output_logits.unsqueeze(0), label_tensor.unsqueeze(0).to(torch.float16))

            loss += crf_loss
    
        return loss / b


import numpy
import math
import cv2
import os
import argparse
from tqdm import tqdm


import torch
import torch.nn as nn

class PSNRLoss(nn.Module):
    def __init__(self, pixel_max=255.0):
        super(PSNRLoss, self).__init__()
        self.pixel_max = pixel_max

    def forward(self, inputs, target):
        # sigmoid -> 0~1사이
        inputs = torch.sigmoid(inputs)
        # MSE 계산
        mse = torch.mean((inputs - target) ** 2)
        # MSE가 0일 경우 처리
        if mse == 0:
            return torch.tensor(0.0, device=inputs.device)  # PSNR Loss는 0
        
        # PSNR 계산
        psnr = 20 * torch.log10(self.pixel_max / torch.sqrt(mse))
        # PSNR을 손실 값으로 변환 (높을수록 좋으므로 반전)
        psnr_loss = 1 - (psnr * 0.01)
        return psnr_loss



class No_20_PSNRLoss(nn.Module):
    def __init__(self, pixel_max=255.0):
        super(No_20_PSNRLoss, self).__init__()
        self.pixel_max = pixel_max

    def forward(self, img1, img2):
        # MSE 계산
        mse = torch.mean((img1 - img2) ** 2)
        # MSE가 0일 경우 처리
        if mse == 0:
            return torch.tensor(0.0, device=img1.device)  # PSNR Loss는 0
        
        # PSNR 계산
        psnr = torch.log10(self.pixel_max / torch.sqrt(mse))
        # PSNR을 손실 값으로 변환 (높을수록 좋으므로 반전)
        psnr_loss = 1 - (psnr * 0.01)
        return psnr_loss

class FocalwithPSNRLoss(nn.Module): # 남겨
    def __init__(self, weight=None, size_average=True):
        super(FocalwithPSNRLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        mse = torch.mean((inputs - targets) ** 2)
        if mse == 0:
            mse = 0.0  # PSNR Loss는 0

        # PSNR 계산
        psnr = 20 * torch.log10(self.pixel_max / torch.sqrt(mse))
        # PSNR을 손실 값으로 변환 (높을수록 좋으므로 반전)
        psnr_loss = 1 - (psnr * 0.01)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs) 
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE         

        total_loss = psnr_loss * focal_loss
        return total_loss


class FocalwithIoULoss(nn.Module): # 남겨
    def __init__(self, weight=None, size_average=True):
        super(FocalwithPSNRLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        mse = torch.mean((inputs - targets) ** 2)
        if mse == 0:
            mse = 0.0  # PSNR Loss는 0

        # PSNR 계산
        psnr = 20 * torch.log10(self.pixel_max / torch.sqrt(mse))
        # PSNR을 손실 값으로 변환 (높을수록 좋으므로 반전)
        psnr_loss = 1 - (psnr * 0.01)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs) 
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE         

        total_loss = psnr_loss * focal_loss
        return total_loss


############################### 누락 losss ##################################
# FocalWithGaussianLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class IoUWithGaussianLoss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, weight=0.1, device=DEVICE):
        super(IoUWithGaussianLoss, self).__init__()
        self.kernel_size = kernel_size
        self.gaussian_kernel = self._create_gaussian_kernel(kernel_size, sigma).to(device)
        self.weight = weight
        self.device = device

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # Create Gaussian kernel
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
        gauss = torch.exp(-0.5 * (x / sigma) ** 2) 
        gauss = gauss / gauss.sum()
        kernel = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)
        return kernel

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)

        iou_loss = 1 - IoU
        # Ensure inputs are 4D for Conv2d
        if inputs.dim() < 4:
            inputs = inputs.unsqueeze(0).unsqueeze(0)

        # Apply Gaussian smoothing
        padding = self.kernel_size // 2
        smoothed_inputs = F.conv2d(inputs, self.gaussian_kernel, padding=padding)
        smoothing_loss = torch.mean((inputs - smoothed_inputs) ** 2)

        # Combine Focal Loss and Smoothing Loss
        total_loss = iou_loss + self.weight * smoothing_loss
        return total_loss


class BCEWithGaussianLoss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, weight=0.1, device=DEVICE):
        super(BCEWithGaussianLoss, self).__init__()
        self.kernel_size = kernel_size
        self.gaussian_kernel = self._create_gaussian_kernel(kernel_size, sigma).to(device)
        self.weight = weight
        self.device = device

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # Create Gaussian kernel
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss = gauss / gauss.sum()
        kernel = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)
        return kernel

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        # Apply Sigmoid if needed
        inputs = torch.sigmoid(inputs).to(self.device)
        targets = targets.to(self.device)

        # Flatten inputs for Focal Loss
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')

        # Ensure inputs are 4D for Conv2d
        if inputs.dim() < 4:
            inputs = inputs.unsqueeze(0).unsqueeze(0)

        # Apply Gaussian smoothing
        padding = self.kernel_size // 2
        smoothed_inputs = F.conv2d(inputs, self.gaussian_kernel, padding=padding)
        smoothing_loss = torch.mean((inputs - smoothed_inputs) ** 2)

        # Combine Focal Loss and Smoothing Loss
        total_loss = BCE + self.weight * smoothing_loss
        return total_loss


class DiceWithGaussianLoss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, weight=0.1, device=DEVICE):
        super(DiceWithGaussianLoss, self).__init__()
        self.kernel_size = kernel_size
        self.gaussian_kernel = self._create_gaussian_kernel(kernel_size, sigma).to(device)
        self.weight = weight
        self.device = device

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # Create Gaussian kernel
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss = gauss / gauss.sum()
        kernel = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)
        return kernel

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)

        iou_loss = 1 - IoU
        # Ensure inputs are 4D for Conv2d
        if inputs.dim() < 4:
            inputs = inputs.unsqueeze(0).unsqueeze(0)

        # Apply Gaussian smoothing
        padding = self.kernel_size // 2
        smoothed_inputs = F.conv2d(inputs, self.gaussian_kernel, padding=padding)
        smoothing_loss = torch.mean((inputs - smoothed_inputs) ** 2)

        # Combine Focal Loss and Smoothing Loss
        total_loss = iou_loss + self.weight * smoothing_loss
        return total_loss

sys.path.append("./.conda/envs/image/lib/python3.12/site-packages/")
import pytorch_ssim

class FocalWithSSIMLoss(nn.Module):
    def __init__(self, ssim_weight=0.1):
        super(FocalWithSSIMLoss, self).__init__()
        self.ssim_weight = ssim_weight

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        inputs = torch.sigmoid(inputs) 
        
        # Focal Loss 계산
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE 

        # Compute SSIM Loss (inputs and targets must be 4D tensors)
        ssim_loss = 1 - pytorch_ssim.ssim(inputs, targets)

        # Combine BCE and SSIM Loss
        total_loss = focal_loss + self.ssim_weight * ssim_loss
        return total_loss


# DiceWithContextualLoss
from torchvision import models

class FocalWithContextualLoss(nn.Module):
    def __init__(self, cx_weight=0.1,device = DEVICE):
        super(FocalWithContextualLoss, self).__init__()
        self.cx_weight = cx_weight
        self.device =device
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval().to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        inputs = torch.sigmoid(inputs) 
        
        # Focal Loss 계산
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE 
        
        # Compute Contextual Loss
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs_features = self.feature_extractor(inputs.repeat(1, 3, 1, 1))
        targets_features = self.feature_extractor(targets.repeat(1, 3, 1, 1))
        contextual_loss = nn.functional.mse_loss(inputs_features, targets_features)

        total_loss = focal_loss + self.cx_weight * contextual_loss
        return total_loss


from torchvision import models

class BCEWithContextualLoss(nn.Module):
    def __init__(self, cx_weight=0.1,device = DEVICE):
        super(BCEWithContextualLoss, self).__init__()
        self.cx_weight = cx_weight
        self.device =device
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval().to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs) 
        
        # Focal Loss 계산
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        
        # Compute Contextual Loss
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs_features = self.feature_extractor(inputs.repeat(1, 3, 1, 1))
        targets_features = self.feature_extractor(targets.repeat(1, 3, 1, 1))
        contextual_loss = nn.functional.mse_loss(inputs_features, targets_features)

        total_loss = BCE + self.cx_weight * contextual_loss
        return total_loss


# FocalWithTVLoss
class IoUWithTVLoss_tar(nn.Module):
    def __init__(self, tv_weight=0.1):
        super(IoUWithTVLoss_tar, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1) # &&

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1) # &&

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)
        iou_loss = 1 - IoU
        
        # Total Variation Loss 계산
        dx = torch.abs(inputs[:, :, 1:, :] - targets[:, :, :-1, :]).mean()
        dy = torch.abs(inputs[:, :, :, 1:] - targets[:, :, :, :-1]).mean()
        tv_loss = dx + dy

        # Focal Loss + TV Loss
        total_loss = iou_loss + self.tv_weight * tv_loss
        return total_loss



## WeightedIoULoss
class DiceWeightedLoss(nn.Module):
    def __init__(self, weight_map=None):
        super(DiceWeightedLoss, self).__init__()
        self.weight_map = weight_map

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        if self.weight_map is not None:
            weights = self.weight_map
        else:
            # Compute weights based on gradient magnitude
            dx = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
            dy = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])

            # Align sizes of dx and dy
            dx = dx[:, :, :, :-1]  # Match width with dy
            dy = dy[:, :, :-1, :]  # Match height with dx

            gradient_magnitude = dx + dy
            weights = gradient_magnitude.detach()

            # Pad weights to match target size
            weights = torch.nn.functional.pad(weights, (0, 1, 0, 1))

        # Compute weighted dice Loss
        intersection = ((inputs * targets) * weights).sum(dim=(1, 2, 3))
        total = ((inputs + targets) * weights).sum(dim=(1, 2, 3))
        dice = (2.*intersection + smooth)/(total + smooth)  
        
        weight_dice_loss = 1 - dice.mean()
        
        return weight_dice_loss

        
## WeightedIoULoss
class BCEWeightedLoss(nn.Module):
    def __init__(self, weight_map=None):
        super(BCEWeightedLoss, self).__init__()
        self.weight_map = weight_map

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        if self.weight_map is not None:
            weights = self.weight_map
        else:
            # Compute weights based on gradient magnitude
            dx = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
            dy = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])

            # Align sizes
            dx = dx[:, :, :, :-1]
            dy = dy[:, :, :-1, :]

            gradient_magnitude = dx + dy
            weights = gradient_magnitude.detach()
            weights = torch.nn.functional.pad(weights, (0, 1, 0, 1))
        
        # Compute pixel-wise BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')  # shape: (B, C, H, W)
        
        # Apply weights
        weighted_bce = weights * bce_loss

        # Take mean
        final_loss = weighted_bce.mean()

        return final_loss


## WeightedIoULoss
class WeightedIoULoss(nn.Module):
    def __init__(self, weight_map=None):
        super(WeightedIoULoss, self).__init__()
        self.weight_map = weight_map

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        if self.weight_map is not None:
            weights = self.weight_map
        else:
            # Compute weights based on gradient magnitude
            dx = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
            dy = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])

            # Align sizes of dx and dy
            dx = dx[:, :, :, :-1]  # Match width with dy
            dy = dy[:, :, :-1, :]  # Match height with dx

            gradient_magnitude = dx + dy
            weights = gradient_magnitude.detach()

            # Pad weights to match target size
            weights = torch.nn.functional.pad(weights, (0, 1, 0, 1))

        # Compute weighted IoU Loss
        intersection = ((inputs * targets) * weights).sum(dim=(1, 2, 3))
        total = ((inputs + targets) * weights).sum(dim=(1, 2, 3))
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou.mean()

        return iou_loss

        
# FocalWithEdgeLoss
class FocalWithEdgeLoss(nn.Module):
    def __init__(self, edge_weight=0.1, device=DEVICE):
        super(FocalWithEdgeLoss, self).__init__()
        self.edge_weight = edge_weight
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.device = device
        sobel_kernel = [[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]]
        self.edge_filter.weight = nn.Parameter(torch.tensor(sobel_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device))

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        # Compute Dice Loss
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        # Compute Edge Loss
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs_edge = torch.abs(self.edge_filter(inputs))
        targets_edge = torch.abs(self.edge_filter(targets))
        edge_loss = nn.functional.l1_loss(inputs_edge, targets_edge)

        total_loss = dice_loss + self.edge_weight * edge_loss
        return total_loss


# BCEWithEdgeLoss
class BCEWithEdgeLoss(nn.Module):
    def __init__(self, edge_weight=0.1, device=DEVICE):
        super(BCEWithEdgeLoss, self).__init__()
        self.edge_weight = edge_weight
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.device = device
        sobel_kernel = [[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]]
        self.edge_filter.weight = nn.Parameter(torch.tensor(sobel_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device))

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        # #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        # BCE
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        inputs_edge = torch.abs(self.edge_filter(inputs))
        targets_edge = torch.abs(self.edge_filter(targets))
        edge_loss = nn.functional.l1_loss(inputs_edge, targets_edge)

        total_loss = BCE + self.edge_weight * edge_loss
        return total_loss

class IoUWeightedLoss(nn.Module):
    def __init__(self, weight_map=None):
        super(IoUWeightedLoss, self).__init__()
        self.weight_map = weight_map

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)

        if self.weight_map is not None:
            weights = self.weight_map
        else:
            # Compute weights based on gradient magnitude
            dx = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
            dy = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])

            # Align sizes
            dx = dx[:, :, :, :-1]
            dy = dy[:, :, :-1, :]

            gradient_magnitude = dx + dy
            weights = gradient_magnitude.detach()
            weights = torch.nn.functional.pad(weights, (0, 1, 0, 1))

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1) # &&

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        IoU = (intersection + smooth)/(total-intersection + smooth)

        iou_loss = 1 - IoU
        
        # Apply weights
        weighted_iou = weights * iou_loss

        # Take mean
        final_loss = weighted_iou.mean()

        return final_loss

########################################여기까지 누락 loss
def psnr(img1, img2):
    mse = numpy.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))