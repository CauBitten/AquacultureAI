import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class FishDataset(Dataset):
    def __init__(self, df, augm=None, mask_augm=None, classification = False, img_size = 128):
        self.dfs = df[['image', 'mask']]
        self.dfys = df[['label', 'label_id']]
        self.augm = augm
        self.mask_augm = mask_augm
        self.classification = classification
        self.img_size = img_size

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, idx):
        label = self.dfys.iloc[idx]['label_id']
        file_image = self.dfs.iloc[idx]['image']
        file_mask = self.dfs.iloc[idx]['mask']

        img = cv2.imread(file_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(file_mask, 0)
        mask = np.where(mask>0, 1.0, 0.0).astype(np.float32)

        if self.augm is not None:
            img  = self.augm(img)

        if self.mask_augm is not None:
            mask = self.mask_augm(mask)

        if self.classification:
            mask_tensor = torch.zeros((9, self.img_size, self.img_size))
            mask_tensor[label] = mask
            mask = mask_tensor


        return img, mask, label