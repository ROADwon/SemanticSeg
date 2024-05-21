import pandas as pd
import os
import torch

from preprocessing import PreProcessing
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader



class LoadCamVid(Dataset):
    def __init__(self, csv_file, label_dir, img_dir, device, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.csv_file = csv_file
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.class_data = pd.read_csv(csv_file)
        self.dict = {}
        self.img_files = os.listdir(self.img_dir)
        self.label_files = os.listdir(self.label_dir)
        
        df = pd.read_csv(self.csv_file)
        
        self.img_files.sort()
        self.label_files.sort()
        
        for i in range(len(df)):
            t = torch.tensor((df.iloc[i,1], df.iloc[i,2], df.iloc[i,3]))
            self.dict[df.iloc[i,0]] = t
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        label = read_image(os.path.join(self.label_dir, self.label_files[idx]))
        onehot = PreProcessing.one_hot_from_label(self.dict, label)
        img = read_image(os.path.join(self.img_dir, self.img_files[idx]))
        
        img = self.transform(img).to(self.device)
        onehot = self.target_transform(onehot).to(self.device)
        label = self.target_transform(label).to(self.device)
        
        return (img, onehot, label)