import torch
import torch.nn as nn

class SoftArgmax(nn.Module):
	def __init__(self, sample_type='norm'):
		super(SoftArgmax, self).__init__()
		self.sample_type = sample_type
	
	def forward(self, x):
		shape = x.shape
		
		if x.dim() == 4:
			# (B, C, M, N) -> (B, C, N)
			w_x = torch.arange(shape[2]).unsqueeze(1).expand_as(x).to(x.device).type(torch.float32)
		elif x.dim() == 5:
			# (B, C, M, N, A) -> (B, C, N, A)
			w_x = torch.arange(shape[2]).unsqueeze(1).unsqueeze(1).expand_as(x).to(x.device).type(torch.float32)
		else:
			raise TypeError
		
		if self.sample_type == 'norm' or not self.training:
			coord_x = x * w_x
		elif self.sample_type == 'uniform':
			eps_x = torch.rand_like(w_x) - 0.5
			coord_x = x * (w_x + eps_x)
		elif self.sample_type == 'triangle':
			eps_x, p_eps_x = uni2tri(torch.rand_like(w_x))
			hm_x = x * p_eps_x
			hm_x = hm_x / hm_x.sum(dim=2, keepdim=True)
			coord_x = hm_x * (w_x + eps_x)
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
    tri = torch.where(eps < 0.5, torch.sqrt(2 * eps) - 1, 1 - torch.sqrt(2 - 2 * eps))
    p = torch.where(tri < 0, tri + 1, - tri + 1)
    return tri, p