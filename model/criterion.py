import torch
import torch.nn as nn

def DiceCELoss(output, label, weight):
	# TODO: to generate a mask map
	pass

def CELoss(output, label):
	pass

def L1Loss(output, label):
	pass


class SummaryLoss(nn.Module):
	def __init__(self):
		super(SummaryLoss, self).__init__()
	
	def forward(self, output, labels):
		layer_maps = output.layer_maps
		surface_maps = output.surface_maps
		final_surfaces = output.final_surfaces
		# L_DiceCe = DiceCELoss(layer_maps, labels['mask'])
		L_Ce = CELoss(surface_maps, labels['bds'])
		L_l1 = L1Loss(final_surfaces, labels['bds'])
		return L_Ce + L_l1