from torch import nn
from torch.nn import functional as F

class UNetAttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(AttentionGate, self).__init__()
        
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