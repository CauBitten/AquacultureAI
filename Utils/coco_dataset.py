from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from detectron2.structures import BoxMode
import cv2
import json
import os

def create_sub_mask_annotation(image_file, sub_mask, image_id, annotation_id):
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
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
    #width = max_x - x
    #height = max_y - y
    bbox = (x, y, max_x, max_y)
    area = multi_poly.area
    height, width = sub_mask.shape[:2] 

    obj = {
        'bbox': bbox,
        'bbox_mode': BoxMode.XYXY_ABS,
        'segmentation': segmentations,
        'category_id': 0,
    }
    
    annotation = {
        'file_name': image_file,
        'width': width,
        'height': height,
        'annotations': [obj],
        'iscrowd': 0,
        'image_id': image_id,
        'id': annotation_id,
        'area': area,
        'mask': sub_mask
    }

    return annotation


def get_annotations(df):
    mask_files = list(df['mask'])
    image_files = list(df['image'])
    annotations = []

    for i in range(len(mask_files)):
        image = image_files[i]
        mask = cv2.imread(mask_files[i], 0)
        mask = np.where(mask>0, 1.0, 0.0).astype(np.float32)
        annotation = create_sub_mask_annotation(image, mask, i+1, i+1)
        annotations.append(annotation)
    
    return annotations