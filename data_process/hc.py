import os
import json
import numpy as np

def json_output(indxs, filename):
	data = []
	for idx in indxs:
		for frame in range(1, 50):
			filename = idx + '_spectralis_macula_v1_s1_R_%d' % (frame)
			fpath = os.path.join(label_path, filename + '.txt')
			ipath = os.path.join(image_path, filename + '.png')
			with open(fpath, 'r') as f:
				dicts = json.loads(f.read())
			bds = dicts['bds']
			mask = dicts['lesion']
			ann = {
				'image_name' : filename + '.png',
				'bds' : bds,
				'mask' : mask
			}
			data.append(ann)
	
	json_path = os.path.join(label_path, filename)
	with open(json_path, 'w') as f:
		f.write(json.dumps(data))

root_path_name = '/Disk1/dataset/OCT/data/hc/'

label_path = os.path.join(root_path_name, 'label/')
image_path = os.path.join(root_path_name, 'image/')

train_idx = ['hc10', 'hc11', 'hc12', 'hc13', 'hc14',
			'ms15', 'ms16', 'ms17', 'ms18', 'ms19', 'ms20', 'ms21']

valid_idx = ['hc09', 'ms13', 'ms14']

test_idx = ['hc01', 'hc02', 'hc03', 'hc04', 'hc05', 'hc06', 'hc07', 'hc08',
			'ms01', 'ms02', 'ms03', 'ms04', 'ms05', 'ms06', 'ms07', 'ms08', 'ms09', 'ms10', 'ms11', 'ms12']

json_output(train_idx, 'train.json')
json_output(valid_idx, 'valid.json')
json_output(test_idx, 'test.json')