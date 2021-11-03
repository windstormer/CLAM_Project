import torch
from torch.utils.data import Dataset
import numpy as np
import os, glob
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle


class CDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data = data_list
        self.transform = transforms.ToTensor()
        self.label_list = torch.FloatTensor(label_list).unsqueeze(1)
    def __getitem__(self, index):
        path = self.data[index]
        img = Image.open(path)
        img = img.convert(mode='RGB')
        tensor = self.transform(img)
        label = self.label_list[index]
        return tensor, label
    
    def __len__(self):
        return len(self.data)


# class VDataset(Dataset):
#     def __init__(self, data_list, label_list):
#         self.data = data_list
#         self.transform = transforms.ToTensor()
#         self.label_list = torch.FloatTensor(label_list).unsqueeze(1)
        
#     def __getitem__(self, index):
#         path = self.data[index]
#         img = Image.open(path)
#         img = img.convert(mode='RGB')
#         tensor = self.transform(img)
#         label = self.label_list[index]
#         return tensor, label
    
#     def __len__(self):
#         return len(self.data)