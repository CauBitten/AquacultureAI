from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import torch
from torch.utils.data import Dataset
import cv2


class FishDataset(Dataset):
    def __init__(self, df, augm=None, mask_augm=None, classification = False, img_size = 128):
        self.dfs = df[['image', 'mask']]
        self.dfys = df[['label', 'label_id']]
        self.augm = augm
        self.mask_augm = mask_augm
        self.classification = classification
        self.img_size = img_size
        self.annotation = create_sub_mask_annotation

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
            mask = self.annotation(mask)

        if self.classification:
            mask_tensor = torch.zeros((9, self.img_size, self.img_size))
            mask_tensor[label] = mask
            mask = mask_tensor


        return img, mask, label


def create_sub_mask_annotation(mask, num_objs = 1):
        contours = measure.find_contours(mask, 0.5, positive_orientation='low')
        segmentations = []
        polygons = []
        for contour in contours:
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=True)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)

        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        bbox = (x, y, max_x, max_y)
        area = multi_poly.area

        data = {}
        data["boxes"] =  bbox
        data["area"] = area
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   
        data["masks"] = mask

        return data

