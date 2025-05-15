#!/usr/bin/env python
# coding: utf-8


import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import itertools
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import heapq
from scipy.optimize import linear_sum_assignment


def get_grad(img):
    sobel_x = torch.tensor([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    gradient_x = F.conv2d(img, sobel_x.to(img.device), padding=1)
    gradient_y = F.conv2d(img, sobel_y.to(img.device), padding=1)
    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + 1e-6)
    gradient_direction = torch.atan2(gradient_y + 1e-6, gradient_x + 1e-6)
    return gradient_magnitude, gradient_direction


def polar_hist(img, num_bins=120, sigma=1.0):
    batch_size = img.size(0)
    _, direct = get_grad(img)
    flat= direct.view(batch_size, -1)
    flat= torch.rad2deg(flat)
    flat_360 = torch.where(flat < 0, flat+360, flat)
    
    bin_edges = torch.linspace(0, 360, num_bins, device=img.device)
    bin_centers = bin_edges + 180 / num_bins
    
    histograms = torch.zeros(batch_size, num_bins, device=img.device)
    for i in range(num_bins):
            # Count elements in each bin range
            histograms[:, i] = torch.exp(-0.5 * ((flat_360 - bin_centers[i]) ** 2) / sigma ** 2).sum(dim=1) / (flat_360.size(1)+1e-6)
    
    return histograms



def polar_loss(true, A):
    true_polar = polar_hist(true)
    A_polar = polar_hist(A)
    diff_polar = torch.sqrt((true_polar-A_polar)**2+1e-6)
    diff_polar = diff_polar.sum(dim=(-1),keepdim=True)/(diff_polar.shape[-1]+1e-6)
    loss = torch.mean(diff_polar)
    return loss




def KL_Loss(true, A):
    true_grad, _ = get_grad(true)
    A_grad, _ = get_grad(A)
    true_grad2, _ = get_grad(true_grad)
    A_grad2, _ = get_grad(A_grad)
    true_grad2_prob = F.softmax(true_grad2.view(true_grad2.size(0), -1), dim=1)  # Flatten spatial dimensions
    A_grad2_prob = F.softmax(A_grad2.view(A_grad2.size(0), -1), dim=1)
    true_grad2_log_prob = torch.log(true_grad2_prob + 1e-6)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    loss = kl_loss(true_grad2_log_prob,  A_grad2_prob)
    return loss

def mean_iou(y_true, y_pred, smooth=1e-6):
    batch_size = y_true.size(0)  # Get the batch size
    iou_sum = 0.0  # Initialize IoU sum
    
    for i in range(batch_size):
        # Squeeze and threshold for individual samples
        y_true_sample = (y_true[i] > 0.5).float()
        y_pred_sample = (y_pred[i] > 0.5).float()
        
        # Calculate intersection and union
        inter = (y_true_sample * y_pred_sample).sum()
        union = y_true_sample.sum() + y_pred_sample.sum() - inter
        
        # Compute IoU for the sample and add it to the sum
        iou = (inter + smooth) / (union + smooth)
        iou_sum += iou.item()
    
    # Calculate the mean IoU for the batch
    mean_iou = iou_sum / batch_size
    return mean_iou




def direction_loss(true,A):
    true_grad, true_dir = get_grad(true)
    A_grad, A_dir = get_grad(A)
    dir_diff = true_dir - A_dir
    weight = (torch.sigmoid(10*A_grad)-0.5)*3.6+0.2 
    weighted_diff = dir_diff * weight
    weighted_norm = torch.norm(weighted_diff, p=2, dim=(2,3))/(weighted_diff.shape[-1]+1e-6)
    weighted_norm = torch.mean(weighted_norm)/4
    return weighted_norm




def length_loss(true, A):
    true_length, _ = get_grad(true)
    A_length, _ = get_grad(A)
    true_length = true_length.sum(dim = (2,3), keepdim=True)
    A_length = A_length.sum(dim= (2,3), keepdim=True)
    true_avg_len = torch.mean(true_length)
    A_avg_len = torch.mean(A_length) #batch mean을 구한 것
    length = A_avg_len/(true_avg_len+1e-6)
    
    return torch.sqrt((1-length)**2)




