import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage.util import random_noise

class ImageNet_noise_reduction(Dataset):
    """
    Class for noise reduction on ImageNet dataset.
    """
    def __init__(self, cfg, subset='train'):
        data_path = f'../data/ImageNet/{subset}/Data'
        self.im_list = []
        for item in os.listdir(data_path)[::30]:
            for subitem in os.listdir(os.path.join(data_path, item)):
                subitem_path = os.path.join(data_path, item, subitem)
                self.im_list.append(subitem_path)        

        self.cfg = cfg

    def calculate_mean_std(self):
        num_pixels_per_image = np.prod(self.cfg.IMAGE_SHAPE)
        sum_pixels = np.zeros(3)
        print('Calculating mean ...')
        for im_path in tqdm(self.im_list):
            im = cv2.imread(im_path)
            im = cv2.resize(im, self.cfg.IMAGE_SHAPE, interpolation=cv2.INTER_LINEAR)
            sum_pixels += np.sum(im, axis=(0, 1)) / num_pixels_per_image 
        mean = sum_pixels / len(self.im_list)
        
        print('Calculating std ...')
        squared_diff = np.zeros(3)
        for im_path in tqdm(self.im_list):
            im = cv2.imread(im_path)
            im = cv2.resize(im, self.cfg.IMAGE_SHAPE, interpolation=cv2.INTER_LINEAR)
            squared_diff += np.sum(np.square(im - mean), axis=(0, 1)) / num_pixels_per_image
        std = np.sqrt(squared_diff / len(self.im_list))

        return mean, std

    def __len__(self):
        return len(self.im_list)

    def normalise(self, im):
        im -= np.array(self.cfg.DATASET_MEAN)
        im /= np.array(self.cfg.DATASET_STD)
        return im

    def __getitem__(self, idx):

        im_path = self.im_list[idx]
        im = cv2.imread(im_path)
        im = cv2.resize(im, self.cfg.IMAGE_SHAPE, interpolation=cv2.INTER_LINEAR)
        im = np.array(im, dtype=np.float32)
        im = self.normalise(im)
        im = im.transpose(2, 0, 1)

        noise = np.random.randn(3, *self.cfg.IMAGE_SHAPE) * self.cfg.NOISE_STD
        noisy_im = im + noise

        data_dict = {
            'im_noisy' : noisy_im,
            'im_orig' : im,
        }

        return data_dict


class GoPro_deblure(Dataset):
    """
    Class for debluring on GoPro dataset.
    """
    def __init__(self, cfg, subset='train'):
        data_path = f'/mnt/GOPRO_Large/{subset}'
        self.im_sharp_list = []
        for item in os.listdir(data_path):
            for subitem in os.listdir(os.path.join(data_path, item, 'sharp')):
                subitem_path = os.path.join(data_path, item, 'sharp', subitem)
                self.im_sharp_list.append(subitem_path)
        self.cfg = cfg

    def calculate_mean_std(self):
        num_pixels_per_image = np.prod(self.cfg.IMAGE_SHAPE)
        sum_pixels = np.zeros(3)
        print('Calculating mean ...')
        for im_path in tqdm(self.im_sharp_list):
            im = cv2.imread(im_path)
            im = cv2.resize(im, self.cfg.IMAGE_SHAPE, interpolation=cv2.INTER_LINEAR)
            sum_pixels += np.sum(im, axis=(0, 1)) / num_pixels_per_image 
        mean = sum_pixels / len(self.im_sharp_list)
        
        print('Calculating std ...')
        squared_diff = np.zeros(3)
        for im_path in tqdm(self.im_sharp_list):
            im = cv2.imread(im_path)
            im = cv2.resize(im, self.cfg.IMAGE_SHAPE, interpolation=cv2.INTER_LINEAR)
            squared_diff += np.sum(np.square(im - mean), axis=(0, 1)) / num_pixels_per_image
        std = np.sqrt(squared_diff / len(self.im_sharp_list))

        return mean, std

    def __len__(self):
        return len(self.im_sharp_list)

    def normalise(self, im):
        im -= np.array(self.cfg.DATASET_MEAN)
        im /= np.array(self.cfg.DATASET_STD)
        return im

    def get_im(self, im_path):
        im = cv2.imread(im_path)
        im = cv2.resize(im, self.cfg.IMAGE_SHAPE, interpolation=cv2.INTER_LINEAR)
        im = np.array(im, dtype=np.float32)
        im = self.normalise(im)
        im = im.transpose(2, 0, 1)
        return im

    def __getitem__(self, idx):

        im_sharp_path = self.im_sharp_list[idx]
        im_blured_path = self.im_sharp_list[idx].replace('sharp', 'blur')
        im_sharp = self.get_im(im_sharp_path)
        im_blured = self.get_im(im_blured_path)

        data_dict = {
            'im_noisy' : im_blured,
            'im_orig' : im_sharp,
        }

        return data_dict

