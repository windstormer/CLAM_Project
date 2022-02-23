import torch
from torch.utils.data import Dataset
import numpy as np
import os, glob
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
import random

class SDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data = data_list
        self.transform = transforms.ToTensor()
        self.label = label_list

    def __getitem__(self, index):
        path = self.data[index]
        label_path = path.replace("tumor", "seg").replace("jpg","png")
        
        img = Image.open(path)
        # img = img.convert(mode='RGB')
        tensor = self.transform(img)
        if label_path == None:
            label = torch.zeros((1, tensor.shape[1], tensor.shape[2]))
        else:
            img = Image.open(label_path)
            label = self.transform(img)
            label = torch.where(label*4 > 0, 1.0, 0.0)
        return tensor, label.float()
    
    def __len__(self):
        return len(self.data)

class StestDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data = data_list
        self.transform = transforms.ToTensor()
        self.label = label_list

    def __getitem__(self, index):
        path = self.data[index]
        label_path = path.replace("tumor", "seg").replace("jpg","png")
        
        img = Image.open(path)
        # img = img.convert(mode='RGB')
        tensor = self.transform(img)
        if label_path == None:
            label = torch.zeros((1, tensor.shape[1], tensor.shape[2]))
        else:
            img = Image.open(label_path)
            label = self.transform(img)
            label = torch.where(label*4 > 0, 1.0, 0.0)
        return path, tensor, label.float()
    
    def __len__(self):
        return len(self.data)