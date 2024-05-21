import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PreProcessing():
    def one_hot_from_label(class_dict, segmented_img):
        semantic_map = []
        for color in class_dict.values():
            
            color = color.unsqueeze(dim=1).unsqueeze(dim=2)
            equal = torch.eq(color, segmented_img)
            semantic_map.append(torch.all(equal, axis=0))
            
        return torch.stack(semantic_map)
    
    def rev_one_hot(class_dict, one_hot, device):
        device = device
        i = 0
        channels, height, width = one_hot.size()
        rgb_image = torch.zeros(3, height, width).to(device)
        
        for color in class_dict.values():
            rgb_image += torch.mul(one_hot[i].to(device), color.unsqueeze(dim=1).unsqueeze(dim=2).to(device))
            
            i += 1
            
        return rgb_image
    
    def prob_to_one_hot(t, device):
        device = device
        channels, height, width = t.size()
        maximus = torch.argmax(t, dim=0)
        
        one_hot = torch.zeros((channels, height, width)).to(device)
        one_hot[maximus, torch.arange(height).unsqueeze(1), torch.arange(width).unsqueeze(0)] = 1
        
        return one_hot
    
    def dict_from_csv(file_path):
        df = pd.read_csv(file_path)
        mappings = {}
        for i in range(len(df)):
            t = torch.tensor((df.iloc[i,1], df.iloc[i,2], df.iloc[i, 3]))
            mappings[df.iloc[i,0]] = t
        
        return mappings