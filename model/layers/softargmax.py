import torch
import torch.nn as nn

class SoftArgmax(nn.Module):
	def __init__(self):
		super(SoftArgmax, self).__init__()
	
	def forward(self, x):
		shape = x.shape # (B, C, M, N) -> (B, C, N)
		hm_x = x * torch.arange(shape[2]).unsqueeze(1).expand_as(x).to(x.device).type(torch.float32)
		coord = hm_x.sum(dim=2) # (B, C, N)
		return coord