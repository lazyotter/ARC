import json
import os
from pathlib import Path
import copy

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

class Data(object):

	def __init__(self, data_path, type):

		self.data_path = data_path
		self.type = type
		self.data = None
		self.keys = None

	def get_data(self):
		data = {}
		
		if self.type == 'test':
			path = Path(self.data_path+'/test')
		if self.type == 'train':
			path = Path(self.data_path+'/training')
		if self.type == 'eval':
			path = Path(self.data_path+'/evaluation')

		for entry in path.iterdir():
			with open(entry,'r') as f:
				data[entry.stem] = json.loads(f.read())

		self.data = data
		self.keys = list(data.keys())

		return data

	def get_train_test(self, data):
		x_t = [x['input'] for x in data['train']]
		y_t = [x['output'] for x in data['train']]
		x_v = [x['input'] for x in data['test']]
		y_v = [x['output'] for x in data['test']]
		return x_t, y_t, x_v, y_v


	def visualize(self, data):
		cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
		norm = colors.Normalize(vmin=0, vmax=9)
		fig, axs = plt.subplots(len(data['train']) + len(data['test']), 2, figsize=(9,9))
		
		for i, img in enumerate(data['train']):
			axs[i, 0].imshow(img['input'], cmap = cmap, norm = norm)
			axs[i, 0].axis('off')
			axs[i, 0].set_title('Train input')
			axs[i, 1].imshow(img['output'], cmap = cmap, norm = norm)
			axs[i, 1].axis('off')
			axs[i, 1].set_title('Train output')

		# Used for loop but theres only one image
		for i, img in enumerate(data['test']):
			row = i + len(data['train'])
			axs[row, 0].imshow(img['input'], cmap = cmap, norm = norm)
			axs[row, 0].axis('off')
			axs[row, 0].set_title('Test input')
			if len(list(img.keys())) == 2:
				axs[row, 1].imshow(img['output'], cmap = cmap, norm = norm)
				axs[row, 1].set_title('Test output')

			axs[row, 1].axis('off')	
		plt.tight_layout()
		plt.show()

	def pad(self,data):
		data1 = copy.deepcopy(data)

		for img in data1['train']:
			img['input'] = self.get_pad(img['input'])
			img['output'] = self.get_pad(img['output'])

		for img in data1['test']:
			img['input'] = self.get_pad(img['input'])
			if len(list(img.keys())) == 2:
				img['output'] = self.get_pad(img['output'])

		return data1


	def get_pad(self,data):
		left = (30 - len(data[0])) // 2
		right = (left + 1) if (len(data[0]) % 2) else left
		top = (30 - len(data)) // 2
		bottom = (top + 1) if (len(data) % 2) else top
		return np.pad(np.array(data), ((top, bottom), (left, right))).tolist()





		