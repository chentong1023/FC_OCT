from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.dataset import Hc
from torch.utils.data.dataloader import DataLoader
from model.fcoct import FCOCT
import torch
import yaml
import os
from easydict import EasyDict as edict


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


exp_name = 'default'
cfg_file = 'original.yaml'

cfg_path = './configs/'
check_point = './exp/' + exp_name + '-' + cfg_file + '/best_model.pth'
cfg = update_config(os.path.join(cfg_path, cfg_file))

m = FCOCT(None, cfg)
m.load_state_dict(torch.load(
    check_point, map_location='cpu'), strict=False)
m = m.cuda()

valid_dataset = Hc(cfg.DATASET.TEST, train=False)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1, shuffle=False, drop_last=False
)
m.eval()

eval_path = './eval'

valid_loader = tqdm(valid_loader, dynamic_ncols=True)

for i, (inps, labels) in enumerate(valid_loader):
    inps = inps.cuda()
    output = m(inps)

    oup_surface = output.final_surfaces.cpu().squeeze().detach().numpy()
    gt_surface = labels['bds']
    gt_weight = labels['weight']
    img = labels['img'].squeeze().numpy()

    plt.imshow(img)
    plt.axis('off')

    # print(oup_surface.shape)

    C = len(oup_surface)
    for i in range(C):
        x = range(len(oup_surface[i]))
        y = oup_surface[i]
        plt.plot(x, y)
    plt.savefig(os.path.join(eval_path, "%d.png" % (labels['img_idx'])))
    plt.close()
