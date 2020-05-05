import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors

class Data(object):

	def __init__(self,data_path):
		self.data_path = data_path

	def get_data(self):
		test_data = Path(self.data_path+'/test')
		training_data = Path(self.data_path+'/training')
		evaluation_data = Path(self.data_path+'/evaluation')
		test = []	
		train = []
		evaluation = []
		for entry in test_data.iterdir():
			with open(entry,'r') as f:
				test.append(json.loads(f.read()))
				#test[entry] = json.loads(f.read())

		for entry in training_data.iterdir():
			with open(entry,'r') as f:
				train.append(json.loads(f.read()))
				#train[entry] = json.loads(f.read())

		for entry in evaluation_data.iterdir():
			with open(entry,'r') as f:
				evaluation.append(json.loads(f.read()))
				#evaluation[entry] = json.loads(f.read())

		return train, test, evaluation

	def visualize(self, data):
		cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
		norm = colors.Normalize(vmin=0, vmax=9)
		fig, axs = plt.subplots(len(data['train']) + len(data['test']), 2, figsize=(10,10))
		
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

		