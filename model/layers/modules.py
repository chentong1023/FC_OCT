import torch
import torch.nn as nn

class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, padding=1):
		super(ResBlock, self).__init__()
		
		self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
		self.conv_block = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.PReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
		)
		self.prelu = nn.PReLU()
	
	def forward(self, x):
		x1 = self.conv_skip(x)
		x2 = self.conv_block(x1)
		x3 = self.prelu(x1 + x2)
		return x3