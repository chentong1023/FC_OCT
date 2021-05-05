import torch
import torch.nn as nn

class TopologyModule(nn.Module):
	def __init__(self):
		super(TopologyModule, self).__init__()
	
	def forward(self, x):
		_, C, N = x.shape
		cx = torch.zeros(x.shape, requires_grad=True).cuda()
		cx[:,0,:]=x[:,0,:]
		for i in range(1, C):
			relu = nn.ReLU()
			cx[:,i,:] = cx[:,i-1,:] + relu(x[:,i,:]-cx[:,i-1,:])
		return cx