import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
 

def paths_dataset(path):
  dir_list = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
  dict_images = {'image': [], 'mask': [], 'label': [], 'label_id': []}
  for dir in dir_list:
      label = dir.split('/')[-1]
      path_image_dir = dir + '/' + label + '/'
      path_mask_dir = dir + '/' + label + ' GT' + '/'
      for image in os.listdir(path_image_dir):
          image_path = path_image_dir + image
          mask_path = path_mask_dir + image
          dict_images['image'].append(image_path)
          dict_images['mask'].append(mask_path)
          dict_images['label'].append(label)
          dict_images['label_id'].append(get_label_id(label))
  df = pd.DataFrame(dict_images)
  return split_dataframe(df)

def paths_dataset_shrimp(path):
  path_image_dir = path + 'Shrimp' + '/'
  path_mask_dir = path + 'Shrimp GT' + '/'
  label = 'Shrimp'
  dict_images = {'image': [], 'mask': [], 'label': [], 'label_id': []}
  for image in os.listdir(path_image_dir):
      image_path = path_image_dir + image
      mask_path = path_mask_dir + image
      dict_images['image'].append(image_path)
      dict_images['mask'].append(mask_path)
      dict_images['label'].append(label)
      dict_images['label_id'].append(get_label_id(label))
  df = pd.DataFrame(dict_images)
  return split_dataframe(df, False)


def split_dataframe(df, stratify = True):
  if stratify:
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify = df['label'])
    df_train, df_valid = train_test_split(df_train, test_size=0.25, random_state=42, stratify = df_train['label'])
  else:
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_valid = train_test_split(df_train, test_size=0.25, random_state=42)

  return df_train, df_valid, df_test


def get_label_id(label):
    label_dict = {
    "Black Sea Sprat": 0,
    "Sea Bass": 1,
    "Red Mullet": 2,
    "Trout": 3,
    "Striped Red Mullet": 4,
    "Shrimp": 5,
    "Red Sea Bream": 6,
    "Hourse Mackerel": 7,
    "Gilt-Head Bream": 8,
    }
    return label_dict[label]

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