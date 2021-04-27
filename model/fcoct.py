import torch
import torch.nn as nn

from model.layers.modules import ResBlock
from model.layers.resunet import ResUnet
from model.layers.softargmax import SoftArgmax
from model.layers.topology import TopologyModule

from easydict import EasyDict

class FCOCT(nn.modules):
	def __init__(self, opt, cfg):
		super(FCOCT, self).__init__()
		self.opt = opt
		self.cfg = cfg
		self.res_unet = ResUnet()
		self.conv1 = nn.Sequential(
			ResBlock(64, 32),
			nn.Conv2d(32, 11, kernel_size=1)
		)
		self.conv2 = nn.Sequential(
			ResBlock(64, 32),
			nn.Conv2d(32, 9, kernel_size=1)
		)
		self.channel_softmax = nn.Softmax(dim=1)
		self.column_softmax = nn.Softmax(dim=3)
		
		self.soft_argmax = SoftArgmax()
		self.topo = TopologyModule()
	
	def forward(self, x):
		feature = self.res_unet(x)
		c1 = self.conv1(feature)
		c2 = self.conv2(feature)
		
		layer_maps = self.channel_softmax(c1)
		surface_maps = self.column_softmax(c2)
		
		surface_positions = self.soft_argmax(surface_maps)
		final_surfaces = self.topo(surface_positions)
		
		return EasyDict(
			layer_maps=layer_maps,
			surface_maps=surface_maps,
			final_surfaces=final_surfaces,
		)
		