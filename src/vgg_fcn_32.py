import torch
import torch.nn as nn

from torchvision.models import models


class vgg_fcn_32(nn.Module):
    def __init__(self, vggnet, n_classes, device):
        super().__init__()
        self.block1 = vggnet.features
        
        self.block3 = nn.Sequential(
            nn.Conv2d(512, 1024, 1, 1),
            nn.Dropout(),
            nn.Conv2d(1024, 2048, 1,1 ),
            nn.Dropout(),
            nn.Conv2d(2048, 2048, 1,1)
        )
        
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d (2048, 2048, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 3, 2, padding=1, output_padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, 3, 2, padding=1, output_padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, n_classes, 3, 2, padding=1, output_padding=1),
            nn.ReLU()
        )
        
        self.classifier = nn.Conv2d(n_classes, n_classes, 1)
        self.last = nn.Softmax2d()
        
    def forward(self, x):
        x = self.block1
        x = self.block3
        x = self.block4
        
        x = self.classifier(x)
        x = self.last(x)
        
        return x
    
        