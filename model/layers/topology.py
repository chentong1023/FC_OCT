import torch
import torch.nn as nn

class TopologyModule(nn.modules):
	def __init__(self):
		super(TopologyModule, self).__init__()
	
	def forward(self, x):
		_, M, N = x.shape
		for i in range(1, M):
			relu = nn.ReLU()
			x[:,i,:] = x[:,i-1,:] + relu(x[:,i,:]-x[:,i-1,:])
		return x