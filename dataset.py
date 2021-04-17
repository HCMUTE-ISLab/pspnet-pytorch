from __future__ import print_function, division

import os
import re
import cv2
import random
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
NUM_CLASSES = 32
BATCH_SIZE = 32
DROPOUT = 0.3

IMG_PATH = "./data/CamVid/train/"
MASK_PATH = "./data/CamVid/train_labels/"


seed = 42
random.seed = seed
np.random.seed = seed


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans







class Camvid(Dataset):
    """Camvid dataset."""
    def __init__(self, image_path = IMG_PATH, mask_path = MASK_PATH):
        files =os.listdir(IMG_PATH)[0:10]
        colors = []
        for file in files:
            img, seg = self.LoadImage(file, IMG_PATH, MASK_PATH)
            colors.append(seg.reshape(seg.shape[0]*seg.shape[1], 3))
        colors = np.array(colors)
        colors = colors.reshape((colors.shape[0]*colors.shape[1],3))

        self.km = KMeans(32)
        self.km.fit(colors)

        self.image_path = image_path
        self.mask_path = mask_path
        self.filename =os.listdir(self.image_path)
        data = torch.tensor(
            [self.LoadImage(i, IMG_PATH, MASK_PATH) for i in self.filename]
        )
        self.image = torch.tensor(data[:,0,:])
        self.mask = torch.tensor(data[:,1,:])
        self.y_cls =  y_cls = [self.ColorsToClass(seg) for seg in self.mask]

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):
        sample = (
            self.image[idx],
            self.mask[idx],
            self.y_cls[idx]
        )
        return sample

    def LoadImage(self, name, path, seg_path):
        img = plt.imread(os.path.join(path, name))
        name_seg = re.sub('\.', '_L.', name)
        
        img = np.array(Image.open(os.path.join(path, name)).convert('RGB').resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS))
        mask = np.array(Image.open(os.path.join(seg_path, name_seg)).convert('RGB').resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS))
        return img/255.0, mask/255.0

    def ColorsToClass(self, seg):
        y_cls = [0]*32
        s = seg.reshape((seg.shape[0]*seg.shape[1],3))
        s = self.km.predict(s)
        s = np.unique(s)
        for i in s:
            y_cls[i] = 1
            
        return torch.tensor(y_cls).view(1,32)
            
