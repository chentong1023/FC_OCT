import os
import json
import numpy as np

def json_output(indxs, json_name):
	data = []
	for idx in indxs:
		for frame in range(1, 12):
			filename = 'Subject_%02d_%d' % (idx, frame)
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
			# print(np.array(bds).shape) (8, 768)
			data.append(ann)
	
	json_path = os.path.join(label_path, json_name)
	with open(json_path, 'w') as f:
		f.write(json.dumps(data))

root_path_name = './data/dme/'

label_path = os.path.join(root_path_name, 'label/')
image_path = os.path.join(root_path_name, 'image/')

train_idx = range(1, 6)
test_idx = range(6, 11)

json_output(train_idx, 'train.json')
json_output(test_idx, 'test.json')