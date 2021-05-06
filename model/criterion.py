import torch
import torch.nn as nn

def DiceCELoss(output, label, weight):
	# TODO: to generate a mask map
	pass

def CELoss(output, label, weight):
	if output.dim() == 4:
		# (B, C, M, N) , (B, C, N)
		out2 = output * weight.unsqueeze(2).expand_as(output)
		out = out2.permute(0, 2, 1, 3)
		cel = nn.CrossEntropyLoss()
		return cel(out, label)
	else:
		raise TypeError

def L1Loss(output, label, weight):
	if output.dim() == 3:
		# (B, C, N)
		out = output * weight
		SmoothL1 = nn.SmoothL1Loss(size_average=True)
		return SmoothL1(out, label)
	elif output.dim() == 4:
		# (B, C, N, A)
		out = output * weight.unsqueeze(3).expand_as(output)
		lbl = label.unsqueeze(3).expand_as(out)
		SmoothL1 = nn.SmoothL1Loss(size_average=True)
		return SmoothL1(out, lbl)
	else:
		raise TypeError


class SummaryLoss(nn.Module):
	def __init__(self):
		super(SummaryLoss, self).__init__()
	
	def forward(self, output, labels):
		# layer_maps = output.layer_maps
		surface_maps = output.surface_maps
		final_surfaces = output.final_surfaces
		device = surface_maps.device
		# L_DiceCe = DiceCELoss(layer_maps, labels['mask'])
		L_Ce = CELoss(surface_maps,
					labels['bds_int'].cuda().detach(),
					labels['weight'].cuda().detach())
		L_l1 = L1Loss(final_surfaces,
					labels['bds'].cuda().detach(),
					labels['weight'].cuda().detach())
		
		return L_Ce + L_l1, L_Ce, L_l1