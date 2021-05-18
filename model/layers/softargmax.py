import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax(nn.Module):
	def __init__(self, sample_type='norm'):
		super(SoftArgmax, self).__init__()
		self.sample_type = sample_type

	def forward(self, x):
		shape = x.shape

		if x.dim() == 4:
			# (B, C, M, N) -> (B, C, N)
			w_x = torch.arange(shape[2]).unsqueeze(
			    1).expand_as(x).to(x.device).type(torch.float32)
		elif x.dim() == 5:
			# (B, C, M, N, A) -> (B, C, N, A)
			w_x = torch.arange(shape[2]).unsqueeze(1).unsqueeze(
			    1).expand_as(x).to(x.device).type(torch.float32)
		else:
			raise TypeError

		if self.sample_type == 'norm' or not self.training:
			coord_x = x * w_x
		elif self.sample_type == 'uniform':
			eps_x = torch.rand_like(w_x) - 0.5
			coord_x = x * (w_x + eps_x)
		elif self.sample_type == 'gaussian':
			eps_x = torch.randn_like(w_x)
			sigma = 1
			gaussian_p_x = torch.exp(- eps_x ** 2 / (2 * sigma ** 2))
			hm_x = x * gaussian_p_x
			hm_x = hm_x / hm_x.sum(dim=2, keepdim=True)
			coord_x = hm_x * (w_x + eps_x)
		elif self.sample_type == 'triangle':
			eps_x, _ = uni2tri(torch.rand_like(w_x))
			w_x = w_x + eps_x
			hm_x = retrive_p(x, w_x)
			hm_x = hm_x / hm_x.sum(dim=2, keepdim=True)
			coord_x = hm_x * w_x
		else:
			raise NotImplementedError
		coord = coord_x.sum(dim=2)
		return coord


def uni2tri(eps):
    # eps U[0, 1]
    # PDF:
    # y = x + 1 (-1 < x < 0)
    # y = -x + 1 (0 < x < 1)
    # CDF:
    # y = x^2 / 2 + x + 1/2 (-1 < x < 0)
    # y = -x^2 / 2 + x + 1/2 (0 < x < 1)
    # invcdf:
    # x = sqrt(2y) - 1, y < 0.5
    # x = 1 - sqrt(2 - 2y), y > 0.5
    tri = torch.where(eps < 0.5, torch.sqrt(2 * eps) -
                      1, 1 - torch.sqrt(2 - 2 * eps))
    p = torch.where(tri < 0, tri + 1, - tri + 1)
    return tri, p


def retrive_p(hm, x):
    # hm: (B, K, W) or (B, K, S, W)
    # x:  (B, K, W) or (B, K, S, W)
	# hm: (B, C, M, N) or (B, C, M, N, A)
	# x:  (B, C, M, N) or (B, C, M, N, A)
	if x.dim() == 4:
		hm = hm.permute(0, 1, 3, 2)
		x = x.permute(0, 1, 3, 2)
	elif x.dim() == 5:
		hm = hm.permute(0, 1, 4, 3, 2)
		x = x.permute(0, 1, 4, 3, 2)
	else:
		raise TypeError

	left_x = x.floor() + 1
	right_x = (x + 1).floor() + 1
	left_hm = F.pad(hm, (1, 1)).gather(-1, left_x.long())
	right_hm = F.pad(hm, (1, 1)).gather(-1, right_x.long())
	new_hm = left_hm + (right_hm - left_hm) * (x + 1 - left_x)
	
	if x.dim() == 4:
		new_hm = new_hm.permute(0, 1, 3, 2)
	elif x.dim() == 5:
		new_hm = new_hm.permute(0, 1, 4, 3, 2)
	else:
		raise TypeError

	return new_hm