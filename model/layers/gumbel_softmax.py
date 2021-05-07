import torch
import torch.nn as nn

class Gumbel_Softmax(nn.Module):
	def __init__(self, dim, tau=1, num_iter=1):
		super(Gumbel_Softmax, self).__init__()
		self.softmax = nn.Softmax(dim=dim)
		self.tau = tau
		self.num_iter = num_iter
	
	def forward(self, x):
		if self.training:
			if self.num_iter > 1:
				tx = x.unsqueeze(4).repeat((1,1,1,1,self.num_iter))
				noise = -torch.log(-torch.log(torch.rand(tx.shape).cuda()))
				hm_gumbel = tx + noise / self.tau
			else:
				noise = -torch.log(-torch.log(torch.rand(x.shape).cuda()))
				hm_gumbel = x + noise / self.tau
		else:
			hm_gumbel = x
		return self.softmax(hm_gumbel)