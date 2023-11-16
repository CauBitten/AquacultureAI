import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from image_preprocess import preprocess
from FishDataset import FishDataset
 

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

def get_images(df, img_size = 128):
  images_list = []
  masks_list = []
  labels_list = []
  images_paths = list(df['image'])
  masks_paths = list(df['mask'])
  labels_col = list(df['label_id'])
  for i in range(len(images_paths)):
    img, mask = preprocess(images_paths[i], masks_paths[i], img_size)
    images_list.append(img)
    masks_list.append(mask)
    labels_list.append(labels_col[i])

  images = np.array(images_list, dtype = float)
  masks = np.array(masks_list, dtype = float)
  labels = np.array(labels_list, dtype = int)

  return images, masks, labels


def split_dataframe(df):
  df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify = df['label'])
  df_train, df_valid = train_test_split(df_train, test_size=0.25, random_state=42, stratify = df_train['label'])
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