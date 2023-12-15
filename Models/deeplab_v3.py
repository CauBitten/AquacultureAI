from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def DeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet50() #pretrained=True, progress=True
    model.classifier = DeepLabHead(2048, outputchannels)
    return model