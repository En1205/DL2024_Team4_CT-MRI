import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 32, dropout_rate=0.1)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64, dropout_rate=0.2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128, dropout_rate=0.3)
        self.pool3 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = DoubleConv(128, 256, dropout_rate=0.5)

        #Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128, dropout_rate=0.3)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64, dropout_rate=0.2)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32, dropout_rate=0.1)

        # Output layer
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bottleneck = self.bottleneck(self.pool3(d3))

        u3 = self.up3(bottleneck)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(u1)
