import torch
import cv2
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

def training_loop(epochs, model, train_loader, valid_loader, criterion, optimizer, device, dtype):
    train_losses = []
    train_iou_list = []
    valid_losses = []
    valid_iou_list = []

    for e in range(epochs):
        print("Epoch {0} out of {1}".format(e+1, epochs))
        
        model, optimizer, train_loss, train_iou = train(model, train_loader, criterion, optimizer, device, dtype)
        train_losses.append(train_loss)
        train_iou_list.append(train_iou)
        
        with torch.no_grad():
            model, valid_loss, valid_iou = valid(model, valid_loader, criterion, device)
            valid_losses.append(valid_loss)
            valid_iou_list.append(valid_iou)
        
        print(f'Train loss: {train_loss:.4f}\t'
            f'Valid loss: {valid_loss:.4f}\t'
            f'Train IoU: {100 * train_iou:.2f}\t'
            f'Valid IoU: {100 * valid_iou:.2f}')
    
    return model, optimizer, train_losses, valid_losses, train_iou_list, valid_iou_list

def train(model, train_loader, criterion, optimizer, device, dtype):
    model.train()
    running_loss = 0
    running_iou = 0
    
    for i, (x, y, z) in enumerate(tqdm(train_loader)):
        x = x.to(device, dtype=dtype)
        x = x.to(device)
        y = [create_sub_mask_annotation(m) for m in y]
        y = y.swapaxes(1, 3).swapaxes(2, 3)
        y.to(device)
        loss_dict = model(x, y)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        print(loss_dict)
        #optimizer.zero_grad()
        
        with torch.no_grad():
            running_loss += loss.item()
            y_np = y.cpu().numpy()
            pred_np = pred.cpu().numpy()
            pred_np = (pred_np > 0.5) * 255
            running_iou += iou_np(y_np, pred_np)

    epoch_loss = running_loss / (i + 1)
    epoch_iou = running_iou / (i + 1)
    
    return model, optimizer, epoch_loss, epoch_iou

def valid(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0
    running_iou = 0

    for i, (x, y, z) in enumerate(valid_loader):

        x = x.to(device)
        y = [create_sub_mask_annotation(m) for m in y]
        y = y.swapaxes(1, 3).swapaxes(2, 3)
        y.to(device)
        loss_dict = model(x, y)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses
        print(loss_dict)
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        pred_np = (pred_np > 0.5) * 255
        running_iou += iou_np(y_np, pred_np)

    epoch_loss = running_loss / (i + 1)
    epoch_iou = running_iou / (i + 1)
    
    return model, epoch_loss, epoch_iou


def iou_np(mask, pred):
    intersection = np.logical_and(mask, pred)
    union = np.logical_or(mask, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def predict(model, image, img_size, threshold, device):
    with torch.no_grad():
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = cv2.resize(image, (img_size, img_size))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image_torch = torch.from_numpy(image).to(device)
        #predMask = model(image_torch).squeeze()
        predMask = model(image_torch)
        if deeplab:
            predMask = predMask['out']
        if rcnn:
            y = create_sub_mask_annotation(y)
        predMask = predMask.squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        predMask = (predMask > threshold) * 255
        predMask = predMask.astype(np.uint8)
        
        return predMask


def create_sub_mask_annotation(mask, num_objs = 1):
        print(mask.shape)
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