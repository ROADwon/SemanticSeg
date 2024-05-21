import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd

import random
import pickle
import os

import torch
import torch.nn as nn
import torchvision.models as models

from abc import ABCMeta
from glob import glob
from tqdm import tqdm
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.io import read_image
from torchsummary import summary

from LoadCamVid import LoadCamVid
from vgg_fcn_8 import vgg_fcn_8
from preprocessing import PreProcessing
from evaluation import Evaluation

vgg = models.vgg16(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)


class ToFloatTensor(object):
    def __call__(self, img):
        return torch.tensor(img).float()
    
trans = transforms.Compose([ToFloatTensor(),
                           transforms.Resize((320, 480))])
target_trans = transforms.Compose([ToFloatTensor(),
                           transforms.Resize((320, 480),
                           interpolation= InterpolationMode.NEAREST_EXACT)])
## dataset load
training_data = LoadCamVid("C:/Users/line/Desktop/Segmentation/data/CamVid/class_dict.csv", "C:/Users/line/Desktop/Segmentation/data/CamVid/train_labels", "C:/Users/line/Desktop/Segmentation/data/CamVid/train",
                           transform=trans, target_transform=target_trans, device=device)
validation_data = LoadCamVid("C:/Users/line/Desktop/Segmentation/data/CamVid/class_dict.csv","C:/Users/line/Desktop/Segmentation/data/CamVid/val_labels", "C:/Users/line/Desktop/Segmentation/data/CamVid/val",
                             transform= trans, target_transform=target_trans, device=device)

## config result save path
save_dir = "./results/pth"
os.makedirs(save_dir, exist_ok=True)

partial_set = torch.utils.data.Subset(training_data, range(4))

train_dataloader = DataLoader(training_data)
validation_dataloader = DataLoader(validation_data)

imgs = set(os.path.splitext(file)[0] for file in validation_data.img_files)
labels = set(os.path.splitext(file)[0] for file in validation_data.label_files)

df = pd.read_csv("../data/CamVid/class_dict.csv")
mappings = {}

for i in range(len(df)):
    t = torch.tensor((df.iloc[i,1], df.iloc[i,2], df.iloc[i,3]))
    mappings[df.iloc[i,0]] = t
    
fcn = vgg_fcn_8(vgg, 32)
fcn.to(device)
print(summary(fcn, input_size=(3, 320, 480)))

layer_wise_gradient = {}
for name, param in fcn.named_parameters():
    if("weight" in name):
        layer_wise_gradient[name] = []
        
epochs = 100
epoch_cnt = []
train_loss =[]
val_loss =[]
val_pixelwise_acc = []
val_mean_acc = []
val_iou = []

torch.cuda.empty_cache()
loss_fnc = nn.BCELoss()
optim = torch.optim.Adam(fcn.parameters(), lr = 0.0001)
scheduler = ExponentialLR(optim, gamma=.9)

for epoch in tqdm(range(epochs)) :
    train_loss.append(0)
    epoch_cnt.append(0)
    val_loss.append(0)
    val_pixelwise_acc.append(0)
    val_mean_acc.append(0)
    val_iou.append(0)
    clip_value = 1
    
    for val_batch_idx, (img, one_hot, label) in enumerate(validation_dataloader):
        fcn.eval()
        with torch.no_grad():
            
            img = img.to(device)
            one_hot = one_hot.to(device)
            label = label.to(device)
            
            output = fcn(img)
            
            val_loss[epoch] += loss_fnc(output, one_hot).item()
            
            for i in range(len(output)):
                j = random.randint(0, len(output) - 1)
                
                one_hot_output = PreProcessing.prob_to_one_hot(output[j], device= device)
                output_label = PreProcessing.rev_one_hot(mappings, one_hot_output, device= device)
                
                val_iou[epoch] += Evaluation.intersection_over_union(one_hot[j], one_hot_output)[1]
                val_pixelwise_acc[epoch] += Evaluation.pixel_acc(label[j], output_label)
                val_mean_acc[epoch] += Evaluation.mean_acc(one_hot[j], one_hot_output)
                
    val_iou[epoch] /= 100
    val_pixelwise_acc  = [acc / 100 for acc in val_pixelwise_acc ]
    val_mean_acc = [acc/100 for acc in val_mean_acc]
    
    for batch_idx, (img, one_hot, label) in enumerate(validation_dataloader):
        fcn.train()
        torch.cuda.empty_cache()
        img = img.to(device)
        one_hot = one_hot.to(device)
        output = fcn(img)
        
        loss_val = loss_fnc(output, one_hot)
        
        train_loss[epoch] += loss_val.item()
        optim.zero_grad()
        loss_val.backward()
        
        if epoch %5 ==0:
            for name,param in fcn.named_parameters():
                if("weight" in name):
                    layer_wise_gradient[name].append(param.grad.norm().item())
                    
        torch.nn.utils.clip_grad_norm_(fcn.parameters(), clip_value)
        optim.step()
        
    scheduler.step()
    train_loss[epoch] = train_loss[epoch]/369
    val_loss[epoch] = val_loss[epoch]/100
    
## save fcn model state every 5 epochs
    if epoch%5 ==0:
        os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(fcn.state_dict(), "fcn8_" + str(epoch) + ".pth")
            
    print(f"Learning Rate : {optim.param_groups[0]['lr']}")
    print(f"epoch : {epoch_cnt[epoch]}, training_loss : {train_loss[epoch]}, validation_loss : {val_loss[epoch]}")
        
## figure graphs


figure_path = save_dir + "Figure"
os.makedirs(figure_path, exist_ok=True)

for index, (key, values) in enumerate(layer_wise_gradient.items()):
    plt.plot(values, label=key)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("LineGraph for each key")
plt.legend()
plt.savefig(os.path.join(figure_path, "linegraph.png"))
plt.close()

plt.plot(epoch_cnt, train_loss, label="train_loss")
plt.plot(epoch_cnt, val_loss, label="val_loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(figure_path, "TrainingLoss.png"))
plt.close()

plt.plot(epoch_cnt, val_iou, label="val_iou")
plt.title("IOU over Epochs")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("IOU")
plt.grid(True)
plt.savefig(os.path.join(figure_path, "IOU.png"))
plt.close()

plt.plot(epoch_cnt, val_pixelwise_acc, label="val_pixelwise_accuracy")
plt.title("pixelwise_acc over Epochs")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("pixelwise_acc")
plt.grid(True)
plt.savefig(os.path.join(figure_path, "Pixelwise.png"))
plt.close()

## model save
model_save_path = os.path.join(save_dir, "model.pt")
torch.save(fcn, model_save_path)