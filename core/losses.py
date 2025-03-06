import numpy as np
import torch
import torch.nn.functional as F


def lm_loss(pred_lms, gt_lms, weight, img_size=224):
    loss = torch.sum(torch.square(pred_lms/img_size - gt_lms /
                                  img_size), dim=2) * weight.reshape(1, -1)
    loss = torch.mean(loss.sum(1))

    return loss


def get_l2(tensor, init_tensor=None):
    if init_tensor is None:
        init_tensor = torch.zeros_like(tensor)
    squared_sum = torch.square(tensor-init_tensor).sum()
    num_elements = tensor.numel()
    normalized_l2 = squared_sum / num_elements
    return normalized_l2
