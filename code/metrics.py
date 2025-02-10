
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import reconstruct_im

class MSE:
    def __init__(self, cfg):
        self.name = "ImMSE"
        self.cfg = cfg

    def calculate(self, pred, label):
        return np.mean((pred - label) ** 2)

class PSNR:
    def __init__(self, cfg):
        self.name = "PSNR"
        self.cfg = cfg

    def calculate(self, pred, label):
        mse = np.sqrt((pred - label) ** 2).mean()
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

class SSIM:
    def __init__(self, cfg):
        self.name = "SSIM"
        self.cfg = cfg

    def calculate(self, pred, label):
        return ssim(pred, label, win_size=7, channel_axis=2, data_range=label.max() - label.min())

def create_metrcs_list(cfg):
    metrcs_list = [MSE(cfg), PSNR(cfg), SSIM(cfg)]
    return metrcs_list

def get_metrics_dict(metrcs_list, batch_dict):
    '''
        Возвращает словарь 
    '''
    metrics_dict = {}
    cfg = metrcs_list[0].cfg
    im_orig_reconstructed = reconstruct_im(batch_dict['im_orig'][0], cfg)

    for number, output in enumerate(batch_dict['outputs']):
        output_reconstructed = reconstruct_im(output[0], cfg)
        for metric in metrcs_list:
            metrics_dict[f'{metric.name} Block {number}' ] = metric.calculate(output_reconstructed, im_orig_reconstructed)
    
    return metrics_dict


