
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim

class MSE:
    def __init__(self):
        self.name = "MSE"

    def calculate(self, pred, label):
        return np.mean((pred - label) ** 2)

class PSNR:
    def __init__(self):
        self.name = "PSNR"

    def calculate(self, pred, label):
        mse = np.mean((pred - label) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

class SSIM:
    def __init__(self):
        self.name = "SSIM"

    def calculate(self, pred, label):
        return ssim(pred, label, win_size=7, channel_axis=2, data_range=label.max() - label.min())


def calc_metrics(pred, label):
    metrics_dict = {}
    for metric in [MSE(), PSNR(), SSIM()]:
        metrics_dict[metric.name] = metric.calculate(pred, label)
    
    return metrics_dict


