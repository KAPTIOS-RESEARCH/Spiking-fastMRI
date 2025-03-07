from src.net.models.unet.vanilla import VanillaUNet, ResUNet
from torchsummary import summary


model = ResUNet()
model.to('cuda')
summary(model, (1, 128, 128))