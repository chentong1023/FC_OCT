import os
import json
import torch

import numpy as numpy

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from utils.metrics import DataLogger
from utils.dataset import Hc

def train(opt, cfg, train_loader, m, criterion, optimizer, writer):
	loss_logger = DataLogger()
	loss_ce_logger = DataLogger()
	loss_l1_logger = DataLogger()
	m.train()
	
	if opt.log:
		train_loader = tqdm(train_loader, dynamic_ncols=True)
	
	for i, (inps, labels) in enumerate(train_loader):
		inps = inps.cuda()
		
		output = m(inps)
		
		
		loss, loss_ce, loss_l1 = criterion(output, labels)
		
		batch_size = inps.shape[0]
		
		loss_logger.update(loss.item(), batch_size)
		loss_ce_logger.update(loss_ce.item(), batch_size)
		loss_l1_logger.update(loss_l1.item(), batch_size)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		opt.trainIters += 1
		
		if opt.log:
            # TQDM
			train_loader.set_description(
				'loss: {loss:.8f} | Cross Entropy: {ce:.4f} | L1 Loss: {l1:.4f}'.format(
                    loss=loss_logger.avg,
                    ce=loss_ce_logger.avg,
					l1=loss_l1_logger.avg)
            )
	if opt.log:
		train_loader.close()

	return loss_logger.avg

def validate(m, opt, cfg, batch_size=2, test=False):
	if test:
		valid_dataset = Hc(cfg.DATASET.TEST, train=False)
	else:
		valid_dataset = Hc(cfg.DATASET.VAL, train=False)
	valid_loader = torch.utils.data.DataLoader(
		valid_dataset, batch_size=batch_size,shuffle=False,drop_last=False
	)
	m.eval()
	
	if opt.log:
		valid_loader = tqdm(valid_loader, dynamic_ncols=True)
	
	mads = []
	
	for i, (inps, labels) in enumerate(valid_loader):
		inps = inps.cuda()
		output = m(inps)
		
		oup_surface = output.final_surfaces.cpu()
		gt_surface = labels['bds']
		gt_weight = labels['weight']
		
		mad = (torch.abs(oup_surface - gt_surface) * gt_weight).sum() / gt_weight.sum()
		mads.append(mad)
	
	return sum(mads) / len(mads)