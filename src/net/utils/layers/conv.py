import torch
import torch.nn.functional as F
from torch import nn
from .regularization import ICLayer

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class UpsampleDSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleDSConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(in_channels, out_channels),
        )
        
    def forward(self, x):
        return self.up(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleDSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleDSConv, self).__init__()
        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ICLayer(out_channels),
            
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ICLayer(out_channels)
        )

    def forward(self, x):
        return self.double_ds_conv(x)
    
    

class DoubleConvBlock(nn.Module):
    """
    Two-layer convolution block with InstanceNorm, LeakyReLU, and Dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, drop_prob: float):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransposeConvBlock(nn.Module):
    """
    Transpose convolution block for upsampling with InstanceNorm and LeakyReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, kernel_size=3, padding=1):
        """
        Initialize a residual convolutional block with a bottleneck design.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): The stride of the convolution.
            expansion (int): The expansion factor for the bottleneck. Typically set to 4 in ResNet.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding to maintain spatial dimensions.
        """
        super(ResConvBlock, self).__init__()

        # Bottleneck layer: 1x1 convolution to reduce the number of channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // expansion, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels // expansion),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
        # 3x3 convolution (main convolution)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // expansion, out_channels // expansion, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels // expansion),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Bottleneck layer: 1x1 convolution to increase the number of channels back
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // expansion, out_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels),
        )
        
        # Shortcut connection (identity mapping) for matching the dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the residual bottleneck block.
        
        Args:
            x (tensor): Input tensor
        
        Returns:
            tensor: Output tensor after applying the residual convolutional block.
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out
