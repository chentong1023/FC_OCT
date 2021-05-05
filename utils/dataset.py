import os
import cv2
import json
import torch
import numpy as np
import scipy.misc
import torch.utils.data as data

class Hc(data.Dataset):
	def __init__(self, cfg, train=True):
		self.root_path = cfg.ROOT
		self.ann_path = os.path.join(self.root_path, cfg.ANN)
		
		with open(self.ann_path, 'r') as f:
			self._lists = json.loads(f.read())
		
		self._train = train
	
	def _transform(self, img, bds, mask):
		# TODO: augument the data by horizontal flipping and vertical scaling
		C, N = bds.shape
		bds_int = np.floor(bds)
		weight = np.zeros(N)
		for i in range(N):
			flag = True
			for j in range(C):
				flag &= not np.isnan(bds[j,i])
			weight[i] = 1 if flag else 0
		bds = np.nan_to_num(bds)
		bds_int = torch.from_numpy(bds_int).type(torch.long)
		bds = torch.from_numpy(bds).type(torch.float32)
		weight = torch.from_numpy(weight).expand_as(bds).type(torch.float32)
		bds = bds * weight
		return {
			'img': img,
			'bds': bds,
			'mask': mask,
			'bds_int': bds_int,
			'weight': weight
		}
	
	def __getitem__(self, idx):
		item = self._lists[idx]
		img_path = os.path.join(self.root_path, 'image/' + item['image_name'])
		img = scipy.misc.imread(img_path)
		N, M = img.shape
		img = torch.from_numpy(img)
		xaxis = torch.arange(N, dtype=torch.uint8).unsqueeze(1).expand_as(img)
		yaxis = torch.arange(M, dtype=torch.uint8).expand_as(img)
		inp = torch.cat((img.unsqueeze(0), xaxis.unsqueeze(0), yaxis.unsqueeze(0)), dim=0).type(torch.float32)
		
		bds = np.array(item['bds'], dtype=np.float32) + 1
		mask = np.array(item['mask'], dtype=np.float32)
		
		target = self._transform(img, bds, mask)
		return inp, target
	
	def __len__(self):
		return len(self._lists)