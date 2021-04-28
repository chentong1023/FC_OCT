import os
import cv2
import numpy as np
import scipy.misc
import torch.utils.data as data

class Hc(data.Dataset):
	def __init__(self, cfg, train=True)
		self.root_path = cfg.ROOT
		self.ann_path = os.path.join(self.root_path, cfg.ANN)
		
		with open(self.ann_path, 'r') as f:
			self._lists = json.loads(f.read())
		
		self._train = train
	
	def _transform(self, img, bds, mask):
		# TODO: augument the data by horizontal flipping and vertical scaling
		return {'img': img, 'bds': bds, 'mask': mask}
	
	def __getitem__(self, idx):
		item = self._lists[idx]
		img_path = os.path.join(self.root_path, 'image/' + item['image_name'])
		img = scipy.misc.imread(img_path)
		bds = np.array(item['bds'], dtype=np.float)
		mask = np.array(item['mask'], dtype=np.float)
		
		target = _transform(img, bds, mask)
		return img, target
		