from torch import nn
from torch.nn import functional as F

class UNetAttentionGate(nn.Module):
    """UNet Attention Gate"""
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(UNetAttentionGate, self).__init__()
        
        inter_channels = inter_channels or in_channels // 2
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g = self.W_g(g)
        x = self.W_x(x)
        
        psi = F.relu(g + x)
        psi = self.psi(psi)
        attention = self.sigmoid(psi)
        return x * attention
    
class SEBlock(nn.Module):
    """Squeeze-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.shape
        se = self.global_avg_pool(x).view(batch, channels)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(batch, channels, 1, 1)
        return x * se
