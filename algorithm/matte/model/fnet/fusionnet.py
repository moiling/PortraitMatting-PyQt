import torch
import torch.nn as nn

from ..ops import Conv2dIBNormRelu


class FusionNet(nn.Module):

    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1 = Conv2dIBNormRelu(7, 64, 5, stride=1, padding=2)
        self.conv2 = Conv2dIBNormRelu(64, 32, 3, stride=1, padding=1)
        self.conv3 = Conv2dIBNormRelu(32, 16, 3, stride=1, padding=1)
        self.conv4 = Conv2dIBNormRelu(16, 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        pred_matte = torch.sigmoid(x)

        return pred_matte
