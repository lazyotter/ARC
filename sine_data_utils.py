import torch
import torchvision
import numpy as np 

import os
import collections

def load_dataset(dataset_name, subset):
	'''
	Inputs:
		- dataset_path: path to the folder of the datasets, eg. ../datasets/miniImageNet
		- subset: train/val/test
	Outputs:
		- all_classes: a dictionary with
			+ key: name of a class
			+ values of a key: names of datapoints within that class
		- all_data: is also a dictionary with
			+ key: name of a datapoint
			+ value: embedding data of that datapoint
		'''
	all_classes = collections.OrderedDict()
	all_data = collections.OrderedDict()
	dataset_root = '/media/n10/Data/datasets'
	np2tensor = torchvision.transforms.ToTensor()

	subset_path = os.path.join(dataset_root, dataset_name, subset)

	all_class_names = [folder_name for folder_name in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, folder_name))]
	for a_class_name in all_class_names:
		a_class_path = os.path.join(subset_path, a_class_name)
		all_classes[a_class_name] = [datapoint_name for datapoint_name in os.listdir(a_class_path)]
	
		for a_datapoint in all_classes[a_class_name]:
			datapoint_embedding = imageio.imread(os.path.join(a_class_path, a_datapoint))
			all_data[a_datapoint] = np2tensor(datapoint_embedding)
	
	return all_classes, all_data

def get_task_sine_data(data_generator, p_sine, num_training_samples, noise_flag=True):
	x, y, _, _, _ = data_generator.generate_sinusoidal_data(noise_flag=noise_flag)

	x_t = x[:num_training_samples]
	y_t = y[:num_training_samples]

	x_v = x[num_training_samples:]
	y_v = y[num_training_samples:]

	return x_t, y_t, x_v, y_v

def get_num_weights(my_net):
	num_weights = 0
	weight_shape = my_net.get_weight_shape()
	for key in weight_shape.keys():
		num_weights += np.prod(weight_shape[key], dtype=np.int32)
	return num_weights

def get_weights_target_net(w_generated, row_id, w_target_shape):
	w = {}
	temp = 0
	for key in w_target_shape.keys():
		w_temp = w_generated[row_id, temp:(temp + np.prod(w_target_shape[key]))]
		if 'b' in key:
			w[key] = w_temp
		else:
			w[key] = w_temp.view(w_target_shape[key])
		temp += np.prod(w_target_shape[key])
	return w