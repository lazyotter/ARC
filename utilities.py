import torch
import numpy as np
from debugger import printer

import collections

def dense_layer(inputs, weights, layer, activation = 'relu'):
	#the layers start from 1 here, but start from 0 in conv_layer
	weight =weights['w{0:d}'.format(layer)].clone()
	out = torch.nn.functional.linear(
		input=inputs,
		weight=weight,
		bias=weights['b{0:d}'.format(layer)]
	)
	if activation is not None:
		out = torch.nn.functional.relu(out)
	return out

def conv_layer(inputs, weights, layer):
	weight = weights['conv{0:d}'.format(layer + 1)].clone()
	out = torch.nn.functional.conv2d(
				input=inputs,
				weight=weight,
				bias=weights['b{0:d}'.format(layer + 1)],
				stride=1,
				padding=1
			)
	out = torch.nn.functional.relu(out)

	out = torch.nn.functional.max_pool2d(
		input=out,
		kernel_size=2,
		stride=2,
		padding=0
	)
	return out

def conv_transpose_layer(inputs, weights, layer, activation='relu'):
	#layers start from 0
	weight = weights['deconv{0:d}'.format(layer + 1)].clone()
	out = torch.nn.functional.conv_transpose2d(
		input=inputs,
		weight=weight,
		bias=weights['b{0:d}'.format(layer + 1)],
		stride=2,
		padding=1,
		output_padding=1)
	if activation == 'relu':
		out = torch.nn.functional.relu(out)
	elif activation == 'sigmoid':
		out = torch.sigmoid(out)
	return out

class ConvNet(torch.nn.Module):
	'''
	http://arxiv.org/abs/1606.04080.pdf
	:param output_size: dimensionality of the output features (global parameters)
	'''
	def __init__(
		self,
		dim_input=(1,32,32),
		dim_output=None,
		num_filters=[64]*4,
		filter_size=(3, 3),
		bn_flag=True,
		device=torch.device('cpu')
	):
		'''
		Inputs:
			- dim_input = (in_channels, H, W)
			- dim_output = None for metric learning (no fully connected layer at the end)
			- num_filters = a list of numbers of filters
			- filter_size = kernel size (Kh, Kw)
			- device = cpu or gpu
		'''
		#super(ConvNet, self).__init__()
		self.dim_input = dim_input[1:]
		self.filter_size = filter_size
		self.dim_output = dim_output
		self.device = device
		self.bn_flag = bn_flag

		num_channels = [dim_input[0]]
		num_channels.extend(num_filters)
		self.num_channels = num_channels

		# auxiliary functions
		if bn_flag:
			self.bn = {}
			for i in range(len(num_filters)):
				self.bn[i + 1] = torch.nn.BatchNorm2d(
					num_features=num_filters[i],
					eps=0,
					momentum=1,
					affine=False,
					track_running_stats=False
				)
	
	def get_dim_features(self):
		'''
		Get the dimensions at the output of each convolutional block
		'''
		dim_features = []

		dim_features.append(self.dim_input)

		# Convolution
		conv_padding = 1
		conv_stride = 1

		# Maxpooling
		pool_padding = 0
		pool_stride = 2

		for _ in range(len(self.num_channels) - 1):
			h_conv2d = np.floor((dim_features[-1][0] + 2*conv_padding - self.filter_size[0])/conv_stride + 1)
			w_conv2d = np.floor((dim_features[-1][1] + 2*conv_padding - self.filter_size[1])/conv_stride + 1)

			h_pool = np.floor((h_conv2d + 2*pool_padding - 2)/pool_stride + 1)
			w_pool = np.floor((w_conv2d + 2*pool_padding - 2)/pool_stride + 1)
			dim_features.append((h_pool, w_pool))
		return dim_features
	
	def get_weight_shape(self):
		w_shape = collections.OrderedDict()
		w_shape_fc = collections.OrderedDict()
		w_shape_dec = collections.OrderedDict()
		for i in range(len(self.num_channels) - 1):
			w_shape['conv{0:d}'.format(i + 1)] = (self.num_channels[i + 1], self.num_channels[i], self.filter_size[0], self.filter_size[1])
			w_shape['b{0:d}'.format(i + 1)] = self.num_channels[i + 1]
		
		if self.dim_output is not None:
			dim_features = self.get_dim_features()
			w_shape['w{0:d}'.format(len(self.num_channels))] = (self.dim_output, self.num_channels[-1]*np.prod(dim_features[-1], dtype=np.int16))
			w_shape['b{0:d}'.format(len(self.num_channels))] = self.dim_output

		# extra layers for inference and generation. Bad coding but deal with it
		#there are 8 other fully connected layers
		#remove the for loop to deal with changing inputs for the fc layers
		w_shape_fc['w1'] = (256, 259)
		w_shape_fc['b1'] = 256
		w_shape_fc['w2'] = (256, 256)
		w_shape_fc['b2'] = 256
		w_shape_fc['w3'] = (256, 256)
		w_shape_fc['b3'] = 256
		w_shape_fc['w4'] = (256, 256)
		w_shape_fc['b4'] = 256
		w_shape_fc['w5'] = (256, 256)
		w_shape_fc['b5'] = 256
		w_shape_fc['w6'] = (256, 256)
		w_shape_fc['b6'] = 256

		#dense layers for gen_input
		w_shape_fc['w7'] = (512, 259)
		w_shape_fc['b7'] = 512
		w_shape_fc['w8'] = (1024, 512)
		w_shape_fc['b8'] = 1024

		#for i in range(8):
		#	w_shape_fc['w{0:d}'.format(i + 1)] = (self.dim_output, self.num_channels[-1]*np.prod(dim_features[-1], dtype=np.int16))
		#	w_shape_fc['b{0:d}'.format(i + 1)] = self.dim_output

		#there are 4 other deconv layers
		#fix the layer list here
		w_shape_dec['deconv1'] = (256, 128, 3, 3)
		w_shape_dec['b1'] = 128
		w_shape_dec['deconv2'] = (128, 64, 3, 3)
		w_shape_dec['b2'] = 64
		w_shape_dec['deconv3'] = (64, 32, 3, 3)
		w_shape_dec['b3'] = 32
		w_shape_dec['deconv4'] = (32, 1, 3, 3)
		w_shape_dec['b4'] = 1
		#for i in range(4):
		#	w_shape_dec['deconv{0:d}'.format(i + 1)] = (self.num_channels[i + 1], self.num_channels[i], self.filter_size[0], self.filter_size[1])
		#	w_shape_dec['b{0:d}'.format(i + 1)] = self.num_channels[i + 1]			

		
		return w_shape, w_shape_fc, w_shape_dec
	
	def forward(self, x, w, deconv=False, p_dropout=0):
		if not deconv:
			out = x
			for i in range(len(self.num_channels) - 1):
				out = conv_layer(out, w, i)
				
				#if self.bn_flag:
				#	out = self.bn[i + 1](out)
			
			out = torch.flatten(input=out, start_dim=1, end_dim=-1)
			if p_dropout > 0:
				out = torch.nn.functional.dropout(input=out, p=p_dropout)

			if self.dim_output is not None:
				out = dense_layer(out, w, len(self.num_channels))
			return out

		else:
			out = x
			# do -2 because the last layer activation is sigmoid
			for i in range(len(self.num_channels) - 2):
				out = conv_transpose_layer(out, w, i)

			#final layer
			out = conv_transpose_layer(out, w, 3, 'sigmoid')

			return out

def get_num_weights(my_net):
	num_weights = 0
	weight_shape, weight_shape_fc,  weight_shape_dec = my_net.get_weight_shape()
	#change the for loop to account for three sets of weights
	for key1, key2, key3 in zip(weight_shape.keys(), weight_shape_fc.keys(), weight_shape_dec.keys()):
		num_weights += np.prod(weight_shape[key1], dtype=np.int32) + \
		np.prod(weight_shape_fc[key2]) + np.prod(weight_shape_dec[key3])
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

def sample_normal(mu, log_variance, num_samples, device):
	"""
	Generate samples from a parameterized normal distribution.
	:param mu: tf tensor - mean parameter of the distribution.
	:param log_variance: tf tensor - log variance of the distribution.
	:param num_samples: np scalar - number of samples to generate.
	:return: tf tensor - samples from distribution of size num_samples x dim(mu).
	"""
	
	shape = torch.unsqueeze(mu,0)

	#shape = torch.cat((torch.FloatTensor(num_samples), mu), axis=-1)
	if device == torch.device('cuda:0'):
		eps = torch.zeros(shape.size(), device=torch.device('cuda:0')).normal_()
	else:
		eps = torch.zeros(shape.size(), device=torch.device('cpu:0')).normal_()

	clamped_variance = log_variance.clamp(max=88)
	#make sure this is not zero
	std = clamped_variance.mul(0.5).exp_()
	return eps.mul(std).add_(mu)

def gen_input(x, weights, angles):
	"""
	generate input to the deconv layers by going through two linear layers
	and reshaping.
	"""

	out = torch.cat((x, angles), axis=1)

	out = dense_layer(out, weights, layer=7)
	out = dense_layer(out, weights, layer=8)
	out = out.view(len(angles), 256, 2, -1)
	return out