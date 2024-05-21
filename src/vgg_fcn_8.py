import torch
import torch.nn as nn
import torchvision.models as models


class vgg_fcn_8(nn.Module):
    def __init__(self, vggnet, n_classes):
        super().__init__()
        self.pool3 = vggnet.features[:17]
        self.pool4 = vggnet.features[17:24]
        self.pool5 = vggnet.features[24:]
        self.convolutions1x1 = nn.Sequential(
            nn.Conv2d(512, 1024, 1, 1),
            nn.Dropout(),
            nn.Conv2d(1024,2048, 1, 1),
            nn.Dropout(),
            nn.Conv2d(2048, 2048, 1,1),
        )
        self.upsample_conv_7 = nn.Sequential(
            nn.ConvTranspose2d(2048,1024, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 3, 1, padding=1)
        )
        self.upsample_pool_4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 3,1, padding=1)
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(256,1024, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024,512, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, n_classes, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(n_classes, n_classes, 1),
            nn.Softmax2d()
        )
    def forward(self,x):
        pool3_output = self.pool3(x)
        pool4_output = self.pool4(pool3_output)
        pool5_output = self.pool5(pool4_output)
        output_conv_7 = self.convolutions1x1(pool5_output)
        upsample_conv_7 = self.upsample_conv_7(output_conv_7)
        upsample_pool_4 = self.upsample_pool_4(pool4_output)
            
        x = torch.add(pool3_output, torch.add(upsample_conv_7, upsample_pool_4))
        output = self.block1(x)
        return output