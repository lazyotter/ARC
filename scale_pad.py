import torch
import copy
import numpy as np

class Scaler(object):

	def __init__(self, data):
		self.data = data
		self.scale_amount = {'train': [], 'test': []}
		self.pad_amount = {'train': [], 'test': []}
		self.pad_scale_data = self.pad_scale()

	def scale_up(self, data = None):
		if data == None:
			data = self.data

		data = copy.deepcopy(data)

		for traintest in data:
			for instance in data[traintest]:
				scale = 32 // max(len(instance['input'][0]), len(instance['input']), len(instance['output'][0]), len(instance['output']))
				instance['input'] = np.kron(np.array(instance['input']), np.ones((scale, scale))).tolist()
				self.scale_amount[traintest].append(scale)
				if len(list(instance.keys())) == 2:
					instance['output'] = np.kron(np.array(instance['output']), np.ones((scale, scale))).tolist()

		return data

	def pad(self, data = None):
		if data == None:
			 data = self.data 

		data1 = copy.deepcopy(data)

		for img in data1['train']:
			img['input'] = self.get_pad(img['input'], 'train')
			img['output'] = self.get_pad(img['output'], 'train')

		for img in data1['test']:
			img['input'] = self.get_pad(img['input'], 'test')
			if len(list(img.keys())) == 2:
				img['output'] = self.get_pad(img['output'], 'test')

		return data1

	def get_pad(self,data,state):
		left = (32 - len(data[0])) // 2
		right = (left + 1) if (len(data[0]) % 2) else left
		top = (32 - len(data)) // 2
		bottom = (top + 1) if (len(data) % 2) else top

		self.pad_amount[state].append([left, right, top, bottom])
		
		return np.pad(np.array(data), ((top, bottom), (left, right))).tolist()
		
	def pad_scale(self):
		scaled_data = self.scale_up()
		padded_scaled = self.pad(scaled_data)

		return padded_scaled