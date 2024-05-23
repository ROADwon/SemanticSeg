import torch
import torch.nn as nn
import os
import cv2 as cv
import torchvision.transforms as transform
import numpy as np

from PIL import Image
from vgg_fcn_8 import vgg_fcn_8
from evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = "C:/Users/line/Desktop/Segmentation/src/results/pth/model.pt" 
model = torch.load(model_save_path)
model.eval()

def preprocess_image(image_path, device):
    trans = transform.Compose([
        transform.Resize((320,480)),
        transform.ToTensor(),
        transform.Normalize(mean=[0,485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = trans(image).unsqueeze(0).to(device)
    
    return image

def predict_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return output

def visualize_prediction(image_path, prediction, class_colors):
    image = cv.imread(image_path)
    color_mask = np.zeros_like(image)
    
    for class_idx, color in class_colors.items():
        mask = prediction == class_idx
        color_mask[mask] = color
    
    result_image = cv.addWeighted(image, .5, color_mask, .5, 0)
    return result_image

image_path = "C:/Users/line/Desktop/Segmentation/data/Test.jpg"
class_colors = {
    0: [64, 0, 128],    # Car
    1: [64, 64, 0],     # Pedestrian
    2: [0, 0, 192],     # Sidewalk
    3: [128, 64, 128],  # Road
    4: [128, 0, 192],   # LaneMkgsDriv
    5: [64, 192, 0]     # Wall
}

image_tensor = preprocess_image(image_path, device)
prediction = predict_image(model, image_tensor)
result_image = visualize_prediction(image_path, prediction, class_colors)

result_path = os.getcwd() + "/predicted_images/predict.png"
cv.imwrite(result_path, result_image)
print(f"Result Image Saved at : {result_path}")

