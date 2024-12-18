import numpy as np
import torch.utils.data as data
import os

class MyDataset(data.Dataset):
    def __init__(self, cfg, subset='train'):
        assert subset in ['tran, test'], 'train or test!'
        self.cfg = cfg
        self.subset = subset
        self.idxes = self.get_idxes()

        self.data = 

    def get_idxes(self):
        labels_path = self.get_labels_path()
        names = os.listdir(labels_path)
        idxes = [name.split('.')[0] for name in names]
        return idxes

    def get_images_path(self):
        path = self.cfg['DATASET_PATH'] + f'/{self.subset}/images'
        return path

    def get_labels_path(self):
        path = self.cfg['DATASET_PATH'] + f'/{self.subset}/labels'
        return path
    
    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
