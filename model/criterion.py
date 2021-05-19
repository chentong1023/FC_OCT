import torch
import torch.nn as nn
import math

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

class L1JointRegressionVar(nn.Module):
	''' L1 Joint Regression Loss
	'''
	def __init__(self, alpha=1, size_average=True):
		super(L1JointRegressionVar, self).__init__()
		self.size_average = size_average
		self.gt_sigma = 2
		self.alpha = alpha

	def forward(self, output, labels):
		surface_maps = output.surface_maps # (B, C, M, N)
		hm = output.heatmaps # (B, C, M, N)
		final_surfaces = output.final_surfaces # (B, C, N)
		device = surface_maps.device
		gt_bds = labels['bds'].cuda().detach()
		gt_bds_weight = labels['weight'].cuda().detach()
		

		w_x = torch.arange(hm.shape[2], dtype=torch.float32, device=device)  / hm.shape[2] - 0.5
		w_x = w_x.unsqueeze(1).expand_as(hm)

		var_x = (w_x - (final_surfaces.unsqueeze(2).expand_as(w_x) / hm.shape[2] - 0.5)) ** 2
		var_x = (hm * var_x).sum(dim=2)

		gt_var_x = self.gt_sigma ** 2 / ((hm.shape[2]) ** 2)

		loss_var = torch.sum(((var_x - gt_var_x) ** 2) * gt_bds_weight) / gt_bds_weight.shape[0]

		l1_loss = L1Loss(final_surfaces,
					gt_bds,
					gt_bds_weight)

		loss = l1_loss + loss_var * self.alpha
		return loss, loss_var, l1_loss

class L1JointRegressionJS(nn.Module):
	''' L1 Joint Regression Loss
	'''
	def __init__(self, alpha=1, size_average=True):
		super(L1JointRegressionJS, self).__init__()
		self.size_average = size_average
		self.gt_sigma = 2
		self.alpha = alpha
		self.k = math.sqrt(2 * math.pi)

	def _kl(self, p, q):
		kl = p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))
		return kl.sum(dim=2, keepdim=False)

	def _js(self, p, q):
		m = 0.5 * (p + q)
		return 0.5 * self._kl(p, m) + 0.5 * self._kl(q, m)

	def forward(self, output, labels):
		
		surface_maps = output.surface_maps # (B, C, M, N)
		hm = output.heatmaps # (B, C, M, N)
		final_surfaces = output.final_surfaces # (B, C, N)
		device = surface_maps.device
		gt_bds = labels['bds'].cuda().detach()
		gt_bds_weight = labels['weight'].cuda().detach()
		

		w_x = torch.arange(hm.shape[2], dtype=torch.float32, device=device)  / hm.shape[2] - 0.5
		w_x = w_x.unsqueeze(1).expand_as(hm)

		dist_x = (w_x - (final_surfaces.unsqueeze(2).expand_as(w_x) / hm.shape[2] - 0.5)) ** 2
		std_x = self.gt_sigma / hm.shape[2]

		gt_var_x = self.gt_sigma ** 2 / ((hm.shape[2]) ** 2)
		gs_x = torch.exp(- dist_x / (2 * (std_x ** 2))) #/ (self.gt_sigma * self.k)

		loss_reg = (self._js(hm, gs_x) * gt_bds_weight).mean()

		l1_loss = L1Loss(final_surfaces,
					gt_bds,
					gt_bds_weight)

		loss = l1_loss + loss_reg * self.alpha
		return loss, loss_reg, l1_loss


class SummaryLoss(nn.Module):
	def __init__(self, alpha = 1):
		super(SummaryLoss, self).__init__()
		self.alpha = alpha
	
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
		
		return self.alpha * L_Ce + L_l1, L_Ce, L_l1