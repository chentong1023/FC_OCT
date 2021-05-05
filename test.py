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
    m = FCOCT(opt, cfg)
    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(
        opt.checkpoint, map_location='cpu'), strict=False)
    m.cuda()

    with torch.no_grad():
        err = validate(m, opt, cfg, test=True)
        if opt.log:
            logger.info(f'##### gt results: {err} #####')


if __name__ == "__main__":
    main()
