import torch
import torch.nn as nn

class SoftArgmax(nn.Module):
	def __init__(self):
		super(SoftArgmax, self).__init__()
	
	def forward(self, x):
		if x.dim() == 4:
			shape = x.shape # (B, C, M, N) -> (B, C, N)
			hm_x = x * torch.arange(shape[2]).unsqueeze(1).expand_as(x).to(x.device).type(torch.float32)
			coord = hm_x.sum(dim=2) # (B, C, N)
		elif x.dim() == 5:
			shape = x.shape # (B, C, M, N, A) -> (B, C, N, A)
			hm_x = x * torch.arange(shape[2]).unsqueeze(1).unsqueeze(1).expand_as(x).to(x.device).type(torch.float32)
			coord = hm_x.sum(dim=2) # (B, C, N, A)
		else:
			raise TypeError
		return coord