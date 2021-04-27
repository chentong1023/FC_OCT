import os
import json

import numpy as numpy

from tqdm import tqdm
from utils.metrics import DataLogger

def train(opt, cfg, train_loader, m, criterion, optimizer, writer):
	loss_logger = DataLogger()
	acc_logger = DataLogger()
	m.train()
	
	if opt.log:
		train_loader = tqdm(train_loader, dynamic_ncols=True)
	
	for i, (inps, labels) in enumerate(train_loader):
		inps = inps.cuda()
		
		for k, _ in labels.items():
			labels[k] = labels[k].cuda()
		
		output = m(inps)
		
		loss = criterion(output, labels)
		
		# acc = calc_accuracy(output, labels)
		acc = 0
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		opt.trainIters += 1
		
		if opt.log:
            # TQDM
            train_loader.set_description(
                'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                    loss=loss_logger.avg,
                    acc=acc_logger.avg)
            )
	if opt.log:
        train_loader.close()

    return loss_logger.avg, acc_logger.avg