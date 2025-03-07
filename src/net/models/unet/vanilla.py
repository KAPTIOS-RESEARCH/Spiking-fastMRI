import torch
from torch import nn
from torch.nn import functional as F
from src.net.utils.layers.conv import *
from src.net.utils.layers.attention import SEBlock


class VanillaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], drop_prob: float = 0.0, use_attention=False):
        super(VanillaUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.use_attention = use_attention

        # Encoder
        for feature in features:
            self.encoder.append(DoubleConvBlock(in_channels, feature, drop_prob=drop_prob))
            if use_attention:
                self.encoder.append(SEBlock(feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConvBlock(features[-1], features[-1] * 2, drop_prob=drop_prob)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(TransposeConvBlock(feature * 2, feature))
            self.decoder.append(DoubleConvBlock(feature * 2, feature, drop_prob=drop_prob))
            if use_attention:
                self.decoder.append(SEBlock(feature))

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        temp_x = x  # To maintain original shape for debugging

        # Encoder forward
        for layer in self.encoder:
            temp_x = layer(temp_x)
            if isinstance(layer, DoubleConvBlock):  # Store only after DoubleConvBlock
                skip_connections.append(temp_x)
                temp_x = F.avg_pool2d(temp_x, kernel_size=2)  # Downsample

        # Bottleneck
        x = self.bottleneck(temp_x)

        # Decoder forward
        skip_connections = skip_connections[::-1]
        skip_idx = 0

        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, TransposeConvBlock):
                skip_connection = skip_connections[skip_idx]
                skip_idx += 1
                if x.shape != skip_connection.shape:
                    x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat((skip_connection, x), dim=1)

        return self.final_conv(x)

class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], drop_prob: float = 0.0, use_attention: bool = True):
        super(ResUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder
        for feature in features:
            self.encoder.append(ResConvBlock(in_channels, feature, drop_prob=drop_prob))
            if use_attention:
                self.encoder.append(SEBlock(feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ResConvBlock(features[-1], features[-1] * 2, drop_prob=drop_prob)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(TransposeConvBlock(feature * 2, feature))
            self.decoder.append(ResConvBlock(feature * 2, feature, drop_prob=drop_prob))
            if use_attention:
                self.decoder.append(SEBlock(feature))
            
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        temp_x = x 

        # Encoder
        for layer in self.encoder:
            temp_x = layer(temp_x)
            if isinstance(layer, ResConvBlock):
                skip_connections.append(temp_x)
                temp_x = F.avg_pool2d(temp_x, kernel_size=2)
                
        # Bottleneck
        x = self.bottleneck(temp_x)

        # Decoder forward
        skip_connections = skip_connections[::-1]
        skip_idx = 0

        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, TransposeConvBlock):
                skip_connection = skip_connections[skip_idx]
                skip_idx += 1
                if x.shape != skip_connection.shape:
                    x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat((skip_connection, x), dim=1)

        return self.final_conv(x)