"""Script for multi-gpu training."""
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data
from utils.opt import cfg, logger, opt
from utils.metrics import NullWriter
from trainer import train, validate
from model.fcoct import FCOCT
from utils.dataset import Hc
from model.criterion import SummaryLoss
from tensorboardX import SummaryWriter

def _init_fn(worker_id):
    np.random.seed(opt.seed)
    random.seed(opt.seed)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    if opt.seed is not None:
        setup_seed(opt.seed)

    if opt.log:
        cfg_file_name = os.path.basename(opt.cfg)
        filehandler = logging.FileHandler(
            './exp/{}-{}/training.log'.format(opt.exp_id, cfg_file_name))
        streamhandler = logging.StreamHandler()

        logger.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
    else:
        null_writer = NullWriter()
        sys.stdout = null_writer

    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    # Model Initialize
    m = FCOCT(opt, cfg)
    # m._initialize()

    m.cuda()

    criterion = SummaryLoss().cuda()
    
    torch.autograd.set_detect_anomaly(True)

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.LR_WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    if opt.log:
        writer = SummaryWriter('.tensorboard/{}-{}'.format(opt.exp_id, cfg.FILE_NAME))
    else:
        writer = None

    train_dataset = Hc(cfg.DATASET.TRAIN, train=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, worker_init_fn=_init_fn)

    opt.trainIters = 0
    best_err = 999

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss = train(opt, cfg, train_loader, m, criterion, optimizer, writer)
        acc = 0
        logger.epochInfo('Train', opt.epoch, loss, acc)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint
            if opt.log:
                torch.save(m.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
            with torch.no_grad():
                err = validate(m, opt, cfg)
                if opt.log and err <= best_err:
                    best_err = err
                    torch.save(m.state_dict(), './exp/{}-{}/best_model.pth'.format(opt.exp_id, cfg.FILE_NAME))

            logger.info(f'##### Epoch {opt.epoch} | gt results: {err}/{best_err} #####')

if __name__ == "__main__":
    main()
