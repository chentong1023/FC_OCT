import torch
import torch.nn as nn

class TopologyModule(nn.Module):
	def __init__(self):
		super(TopologyModule, self).__init__()
	
	def forward(self, x):
		if x.dim() == 3:
			_, C, N = x.shape
			cx = torch.zeros(x.shape, requires_grad=True).cuda()
			cx[:,0,:]=x[:,0,:]
			for i in range(1, C):
				relu = nn.ReLU()
				cx[:,i,:] = cx[:,i-1,:] + relu(x[:,i,:]-cx[:,i-1,:])
		elif x.dim() == 4:
			_, C, N, _ = x.shape
			cx = torch.zeros(x.shape, requires_grad=True).cuda()
			cx[:,0,:,:]=x[:,0,:,:]
			for i in range(1, C):
				relu = nn.ReLU()
				cx[:,i,:,:] = cx[:,i-1,:,:] + relu(x[:,i,:,:]-cx[:,i-1,:,:])
		else:
			raise TypeError
		return cx