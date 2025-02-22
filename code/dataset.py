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
        for item in os.listdir(data_path)[::10]:
            for subitem in os.listdir(os.path.join(data_path, item)):
                subitem_path = os.path.join(data_path, item, subitem)
                self.im_list.append(subitem_path)        

        self.cfg = cfg

    def calculate_mean_std(self):
        num_pixels_per_image = np.prod(self.cfg.IMAGE_SHAPE)
        sum_pixels = np.zeros(3)
        print('Calculating mean ...')
        for data_dict in tqdm(self):
            sum_pixels += np.sum(data_dict['original_im'], axis=(0, 1)) / num_pixels_per_image 
        mean = sum_pixels / self.__len__()
        
        print('Calculating std ...')
        squared_diff = np.zeros(3)
        for data_dict in tqdm(self):
            squared_diff += np.sum(np.square(data_dict['original_im'] - mean), axis=(0, 1)) / num_pixels_per_image
        std = np.sqrt(squared_diff / self.__len__())

        return mean, std

    def __len__(self):
        return len(self.im_list)

    def normalise(self, im):
        im -= np.array(self.cfg.IMAGENET_MEAN)
        im /= np.array(self.cfg.IMAGENET_STD)
        return im

    def __getitem__(self, idx):
        """
        Возвращает один элемент данных (пару x, y) по индексу.
        """

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
        data_path = f'../data/ImageNet/{subset}'
        self.im_sharp_list = []
        for item in os.listdir(data_path)[::5]:
            for subitem in os.listdir(os.path.join(data_path, item, 'sharp')):
                subitem_path = os.path.join(data_path, item, 'sharp', subitem)
                self.im_sharp_list.append(subitem_path)
        self.cfg = cfg

    def calculate_mean_std(self):
        num_pixels_per_image = np.prod(self.cfg.IMAGE_SHAPE)
        sum_pixels = np.zeros(3)
        print('Calculating mean ...')
        for data_dict in tqdm(self):
            sum_pixels += np.sum(data_dict['original_im'], axis=(0, 1)) / num_pixels_per_image 
        mean = sum_pixels / self.__len__()
        
        print('Calculating std ...')
        squared_diff = np.zeros(3)
        for data_dict in tqdm(self):
            squared_diff += np.sum(np.square(data_dict['original_im'] - mean), axis=(0, 1)) / num_pixels_per_image
        std = np.sqrt(squared_diff / self.__len__())

        return mean, std

    def __len__(self):
        return len(self.im_sharp_list)

    def normalise(self, im):
        im -= np.array(self.cfg.GOPRO_MEAN)
        im /= np.array(self.cfg.GOPRO_STD)
        return im

    def __getitem__(self, idx):

        im_sharp_path = self.im_sharp_list[idx]
        im_sharp_path = self.im_sharp_list[idx]
        im_sharp = cv2.imread(im_path)
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

