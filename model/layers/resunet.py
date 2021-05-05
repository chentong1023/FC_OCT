import torch
import torch.nn as nn
from model.layers.modules import ResBlock

class ResUnet(nn.Module):
	def __init__(self, channel=3, filter=[64, 128]):
		super(ResUnet, self).__init__()
		
		self.resblock1 = ResBlock(channel, filter[0]) # (64, M, N)
		self.resblock2 = ResBlock(filter[0], filter[0])
		
		self.maxpool1 = nn.MaxPool2d(kernel_size=2) # (64, M/2, N/2)
		self.resblock3 = ResBlock(filter[0], filter[0])
		
		self.maxpool2 = nn.MaxPool2d(kernel_size=2) # (64, M/4, N/4)		
		self.resblock4 = ResBlock(filter[0], filter[0])
		
		self.maxpool3 = nn.MaxPool2d(kernel_size=2) # (64, M/8, N/8)
		self.resblock5 = ResBlock(filter[0], filter[0])
		
		self.maxpool4 = nn.MaxPool2d(kernel_size=2) # (64, M/16, N/16)
		self.resblock6 = ResBlock(filter[0], filter[0])
		
		self.upsample4 = nn.Upsample(scale_factor=2)
		self.resblock7 = ResBlock(filter[1], filter[0])
		
		self.upsample3 = nn.Upsample(scale_factor=2)
		self.resblock8 = ResBlock(filter[1], filter[0])
		
		self.upsample2 = nn.Upsample(scale_factor=2)
		self.resblock9 = ResBlock(filter[1], filter[0])
		
		self.upsample1 = nn.Upsample(scale_factor=2)
		self.conv1x1 = nn.Conv2d(filter[1], filter[0], kernel_size=1)
		
	def forward(self, x):
		x1 = self.resblock1(x)
		x2 = self.resblock2(x1)
		
		x3 = self.maxpool1(x2)
		x4 = self.resblock3(x3)
		
		x5 = self.maxpool2(x4)
		x6 = self.resblock4(x5)
		
		x7 = self.maxpool3(x6)
		x8 = self.resblock5(x7)
		
		x9 = self.maxpool4(x8)
		x10 = self.resblock6(x9)
		
		x11 = self.upsample4(x10)
		x12 = self.resblock7(torch.cat([x11, x8], dim=1))
		
		x13 = self.upsample3(x12)
		x14 = self.resblock8(torch.cat([x13, x6], dim=1))
		
		x15 = self.upsample2(x14)
		x16 = self.resblock9(torch.cat([x15, x4], dim=1))
		
		x17 = self.upsample1(x16)
		x18 = self.conv1x1(torch.cat([x17, x2], dim=1))
		
		return x18