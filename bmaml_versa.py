import torch
#import torchvision

import numpy as np 
import random
import itertools
from scipy.special import erf

from DataGeneratorT import DataGenerator
from utilities import ConvNet, get_num_weights, get_weights_target_net, gen_input
from inference import task_inference
from load_data import Data
from scale_pad import Scaler

import os
import sys
import argparse

class Bmaml(object):

	def __init__(self, 
				 k=5, 
				 n_way=1, 
				 train=True,
				 #maybe get rid of test 
				 test=False,
				 inner_lr=1e-3, 
				 num_inner_updates=0,
				 meta_lr=1e-3,
				 minibatch_size=10,
				 num_epochs=100,
				 num_particles=5,
				 lr_decay=1,
				 meta_l2_regularization=0,
				 num_val_tasks=100,
				 p_dropout_base=0,
				 device = torch.device('cuda:0')):

		self.num_training_samples_per_class = k
		self.num_classes_per_task = n_way
		#maybe + 3 for us depending on how this works
		self.num_total_samples_per_class = k + 1
		self.train = train
		self.test = test
		self.inner_lr = inner_lr
		self.num_inner_updates=num_inner_updates
		self.meta_lr = meta_lr
		self.num_tasks_per_minibatch = minibatch_size
		self.num_meta_updates_print = int(1000/minibatch_size)
		self.num_epochs = num_epochs
		self.expected_total_tasks_per_epoch = 10000
		#THIS SEEMS LIKE ITS JUST EXPECTED TOTAL TASKS PER EPOCH
		self.num_tasks_per_epoch = int(self.expected_total_tasks_per_epoch/self.num_tasks_per_minibatch)*self.num_tasks_per_minibatch
		self.num_particles = num_particles
		self.lr_decay = lr_decay
		self.meta_l2_regularization = meta_l2_regularization
		self.num_val_tasks = num_val_tasks
		self.p_dropout_base = p_dropout_base
		self.p_sine = 0.5
		#Change this loss to some other kind of loss
		self.loss = torch.nn.MSELoss()
		#two thetas for two losses
		self.theta_enc = []
		self.theta_am = []
		self.op_theta_enc = None
		self.op_theta_am = None
		self.w_shape = None
		self.w_shape_fc = None
		self.w_shape_dec = None
		self.num_epochs_save = 1
		#THIS SEEMS LIKE ITS JUST EXPECTED TOTAL TASKS PER EPOCH
		expected_tasks_save_loss = 2000
		self.num_tasks_save_loss = int(expected_tasks_save_loss/self.num_tasks_per_minibatch)*self.num_tasks_per_minibatch
		dst_folder_root = './output'
		self.dst_folder = '{0:s}/BMAML_versa_few_shot/BMAML_{1:s}_{2:d}way_{3:d}shot'.format(
			dst_folder_root,
			'arc',
			self.num_classes_per_task,
			self.num_training_samples_per_class
		)
		self.device = device
		#self.device = torch.device('cpu')
		self.net = ConvNet(dim_output=256, device=self.device)
		if not os.path.exists(self.dst_folder):
			os.makedirs(self.dst_folder)
			print('No folder for storage found')
			print('Make folder to store meta-parameters at')
		else:
			print('Found existing folder. Meta-parameters will be stored at')
		print(self.dst_folder)

	def init_theta(self):
		#get three different sets of weights for three different networks
		self.w_shape, self.w_shape_fc, self.w_shape_dec = self.net.get_weight_shape()
		num_weights = get_num_weights(self.net)
		print('Number of parameters of base model = {0:d}'.format(num_weights))
		for _ in range(self.num_particles):
			theta_flatten = []
			#do three for loops for the three sets of keys (bad code)
			for key in self.w_shape.keys():
				if isinstance(self.w_shape[key], tuple):
					theta_temp = torch.empty(self.w_shape[key], device = self.device)
					torch.nn.init.xavier_normal_(tensor=theta_temp)
				else:
					theta_temp = torch.zeros(self.w_shape[key], device = self.device)
				theta_flatten.append(torch.flatten(theta_temp, start_dim=0, end_dim=-1))
			self.theta_enc.append(torch.cat(theta_flatten))

			theta_flatten = []
			for key in self.w_shape_fc.keys():
				if isinstance(self.w_shape_fc[key], tuple):
					theta_temp = torch.empty(self.w_shape_fc[key], device = self.device)
					torch.nn.init.xavier_normal_(tensor=theta_temp)
				else:
					theta_temp = torch.zeros(self.w_shape_fc[key], device = self.device)
				#why end_dim = -1?
				theta_flatten.append(torch.flatten(theta_temp, start_dim=0, end_dim=-1))
			#check to make sure this copy is not being changed
			flattened_tensor = torch.cat(theta_flatten)
			#self.theta_am.append(torch.cat(theta_flatten))

			theta_flatten = []
			for key in self.w_shape_dec.keys():
				if isinstance(self.w_shape_dec[key], tuple):
					theta_temp = torch.empty(self.w_shape_dec[key], device = self.device)
					torch.nn.init.xavier_normal_(tensor=theta_temp)
				else:
					theta_temp = torch.zeros(self.w_shape_dec[key], device = self.device)
				theta_flatten.append(torch.flatten(theta_temp, start_dim=0, end_dim=-1))
			#hopefully this works, it may have messed up the indexing
			self.theta_am.append(torch.cat((flattened_tensor,torch.cat(theta_flatten))))

		self.theta_enc = torch.stack(self.theta_enc)
		self.theta_am = torch.stack(self.theta_am)
		self.theta_enc.requires_grad_()
		self.theta_am.requires_grad_()
		self.op_theta_enc = torch.optim.Adam(
			params=[self.theta_enc],
			lr = self.meta_lr)
		#change lr to another SVDM lr
		self.op_theta_am = torch.optim.Adam(
			params=[self.theta_am],
			lr = self.meta_lr)

	def meta_train(self, train_subset='train'):
		torch.autograd.set_detect_anomaly(True)
		if train_subset == 'train':
			train_data_loader = Data('./data', 'train')
			train_data = train_data_loader.get_data()
			keys = train_data_loader.keys


		print('Start to train...')
		for epoch in range(0, self.num_epochs):
			#variables for monitoring
			meta_loss_saved = []
			enc_loss_saved = []
			val_accuracies = []
			train_accuracies = []

			meta_loss = 0 #accumulate the loss of many ensembling networks
			enc_loss = 0 #accumulate the loss of the encoder network
			num_meta_updates_count = 0

			meta_loss_avg_print = 0
			enc_loss_avg_print = 0
			#meta_mse_avg_print = 0

			meta_loss_avg_save = []
			enc_loss_avg_save = []
			#meta_mse_avg_save = []

			task_count = 0

			#change to maybe do multiple tasks
			for key in keys:
				scaler = Scaler(train_data[key])
				padded_data = scaler.pad_scale()
				x_t, y_t, x_v, y_v = train_data_loader.get_train_test(padded_data, device=self.device)

				chaser, leader, chaser_loss = self.get_task_prediction(x_t, y_t, x_v, y_v)
				loss_NLL = self.get_meta_loss(chaser, leader)
				print(chaser_loss)

				if torch.isnan(loss_NLL).item():
					sys.exit('NaN error')

				if torch.isnan(chaser_loss).item():
					sys.exit('NaN error')

				meta_loss = meta_loss + loss_NLL
				enc_loss = enc_loss + chaser_loss
				#meta_mse = self.loss(y_pred, y_v)

				task_count = task_count + 1

				#maybe get rid of this task count thing
				#if task_count % self.num_tasks_per_minibatch == 0:
				meta_loss = meta_loss/len(x_t)
				enc_loss = enc_loss/len(x_t)
				#meta_mse = meta_mse/self.num_tasks_per_minibatch

				# accumulate into different variables for printing purpose
				meta_loss_avg_print += meta_loss.item()
				enc_loss_avg_print += enc_loss.item()
				#meta_mse_avg_print += meta_mse.item()

				#Hello Aidan, please do this twice for two different loss functions (with an extra zero grad)
				self.op_theta_enc.zero_grad()
				chaser_loss.backward(retain_graph=True)
				self.op_theta_enc.step()

				self.op_theta_am.zero_grad()
				meta_loss.backward()
				self.op_theta_am.step()

				# Printing losses
				num_meta_updates_count += 1
				if (num_meta_updates_count % self.num_meta_updates_print == 0):
					meta_loss_avg_save.append(meta_loss_avg_print/num_meta_updates_count)
					enc_loss_avg_save.append(enc_loss_avg_print/num_meta_updates_count)

					#meta_mse_avg_save.append(meta_mse_avg_print/num_meta_updates_count)
					print('{0:d}, {1:2.4f}, {1:2.4f}'.format(
						task_count,
						meta_loss_avg_save[-1],
						enc_loss_avg_save[-1]
						#meta_mse_avg_save[-1]
					))

					num_meta_updates_count = 0
					meta_loss_avg_print = 0
					enc_loss_avg_print = 0
					#meta_mse_avg_print = 0
				
				if (task_count % self.num_tasks_save_loss == 0):
					meta_loss_saved.append(np.mean(meta_loss_avg_save))
					enc_loss_saved.append(np.mean(enc_loss_avg_save))

					meta_loss_avg_save = []
					enc_loss_avg_save = []
					#meta_mse_avg_save = []

					# print('Saving loss...')
					# val_accs, _ = meta_validation(
					#     datasubset=val_set,
					#     num_val_tasks=num_val_tasks,
					#     return_uncertainty=False)
					# val_acc = np.mean(val_accs)
					# val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
					# print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
					# val_accuracies.append(val_acc)

					# train_accs, _ = meta_validation(
					#     datasubset=train_set,
					#     num_val_tasks=num_val_tasks,
					#     return_uncertainty=False)
					# train_acc = np.mean(train_accs)
					# train_ci95 = 1.96*np.std(train_accs)/np.sqrt(num_val_tasks)
					# print('Train accuracy = {0:2.4f} +/- {1:2.4f}\n'.format(train_acc, train_ci95))
					# train_accuracies.append(train_acc)
				
				# reset meta loss
				meta_loss = 0
				enc_loss = 0

#			if (task_count >= self.num_tasks_per_epoch):
#				break
			if ((epoch + 1)% self.num_epochs_save == 0):
				checkpoint = {
					'theta_am': self.theta_am,
					'theta_enc': self.theta_enc,
					'meta_loss': meta_loss_saved,
					'enc_loss': enc_loss_saved,
					'val_accuracy': val_accuracies,
					'train_accuracy': train_accuracies,
					'op_theta': self.op_theta_am.state_dict()
				}
				print('SAVING WEIGHTS...')
				checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt')\
							.format('sine_line',
									self.num_classes_per_task,
									self.num_training_samples_per_class,
									epoch + 1)
				print(checkpoint_filename)
				torch.save(checkpoint, os.path.join(self.dst_folder, checkpoint_filename))
				print(checkpoint['meta_loss'])
				print(checkpoint['enc_loss'])
			print()

	def get_task_prediction(self, x_t, y_t, x_v, y_v=None, predict=False):
		'''
		If y_v is not None:
			this is training
			return NLL loss
		Else:
			this is testing
			return the predicted labels y_pred_v of x_v
		'''

		#MAYBE DON'T NEED NUM_INNER_UPDATES

		d_NLL_chaser = []
		d_NLL_leader = []
		for particle_id in range(self.num_particles):
			w_enc = get_weights_target_net(w_generated=self.theta_enc, row_id=particle_id, w_target_shape=self.w_shape)
			#chaser

			#extract features
			in_features = self.net.forward(x=x_t, w=w_enc)
			train_length = len(in_features)
			out_features = self.net.forward(x=y_t[:train_length-1], w=w_enc)
			
			#parameters
			w_fc = get_weights_target_net(w_generated=self.theta_am, row_id=particle_id, w_target_shape=self.w_shape_fc)
			task_params = task_inference(in_features[:train_length-1], out_features, weights=w_fc, device=self.device)
			
			#generate image
			sample_log_py = []
			task_vector = task_params['psi_samples'][0, :, :]
			#repeat this for the batch, CHECK IF REPEAT TENSOR IS THE RIGHT SIZE
			task_vector = task_vector.repeat(len(in_features[train_length-1:]), 1)
			task_vector = gen_input(task_vector, w_fc, in_features[train_length-1:])
			w_deconv = get_weights_target_net(w_generated=self.theta_am, row_id=particle_id, w_target_shape=self.w_shape_dec)
			y_pred_t = self.net.forward(x=task_vector, w=w_deconv, deconv=True, p_dropout=self.p_dropout_base,)
			loss_NLL_chaser = self.loss(y_pred_t, y_t[train_length-1:])
	
			#two gradients for w_fc and w_deconv
			NLL_grads_chaser_fc = torch.autograd.grad(
				outputs=loss_NLL_chaser,
				inputs=w_fc.values(),
				create_graph=True
			)
			NLL_gradients_chaser_fc = dict(zip(w_fc.keys(), NLL_grads_chaser_fc))
			NLL_gradients_tensor_chaser_fc = self.dict2tensor(dict_obj=NLL_gradients_chaser_fc)
			#d_NLL_chaser_fc.append(NLL_gradients_tensor_chaser_fc)

			NLL_grads_chaser_deconv = torch.autograd.grad(
				outputs=loss_NLL_chaser,
				inputs=w_deconv.values(),
				create_graph=True
			)
			NLL_gradients_chaser_deconv = dict(zip(w_deconv.keys(), NLL_grads_chaser_deconv))
			NLL_gradients_tensor_chaser_deconv = self.dict2tensor(dict_obj=NLL_gradients_chaser_deconv)
			#d_NLL_chaser_deconv.append(NLL_gradients_tensor_chaser_deconv)

			#should I concatenate or stack?
			d_NLL_chaser.append(torch.cat((NLL_gradients_tensor_chaser_fc,NLL_gradients_tensor_chaser_deconv)))
			#d_NLL_chaser.append(d_NLL_chaser_fc.cat(d_NLL_chaser_deconv))

			#leader
			x = torch.cat((x_t,x_v),0)
			y = torch.cat((y_t,y_v),0)

			#extract features
			in_features = self.net.forward(x=x, w=w_enc)
			out_features = self.net.forward(x=y[:train_length], w=w_enc)
			
			#parameters
			w_fc = get_weights_target_net(w_generated=self.theta_am, row_id=particle_id, w_target_shape=self.w_shape_fc)
			task_params = task_inference(in_features[:train_length], out_features, weights=w_fc, device=self.device)
			
			#generate image
			sample_log_py = []
			task_vector = task_params['psi_samples'][0, :, :]
			#repeat this for the batch, CHECK IF REPEAT TENSOR IS THE RIGHT SIZE
			task_vector = task_vector.repeat(len(in_features[train_length:]), 1)
			task_vector = gen_input(task_vector, w_fc, in_features[train_length:])
			w_deconv = get_weights_target_net(w_generated=self.theta_am, row_id=particle_id, w_target_shape=self.w_shape_dec)

			y_pred = self.net.forward(x=task_vector, w=w_deconv, deconv=True, p_dropout=self.p_dropout_base)
			loss_NLL_leader = self.loss(y_pred, (torch.cat((y_t,y_v),0)[train_length:]))

			#two gradients for w_fc and w_deconv
			NLL_grads_leader_fc = torch.autograd.grad(
				outputs=loss_NLL_leader,
				inputs=w_fc.values(),
				create_graph=True
			)
			NLL_gradients_leader_fc = dict(zip(w_fc.keys(), NLL_grads_leader_fc))
			NLL_gradients_tensor_leader_fc = self.dict2tensor(dict_obj=NLL_gradients_leader_fc)
			#d_NLL_leader_fc.append(NLL_gradients_tensor_leader_fc)

			NLL_grads_leader_deconv = torch.autograd.grad(
				outputs=loss_NLL_leader,
				inputs=w_deconv.values(),
				create_graph=True
			)
			NLL_gradients_leader_deconv = dict(zip(w_deconv.keys(), NLL_grads_leader_deconv))
			NLL_gradients_tensor_leader_deconv = self.dict2tensor(dict_obj=NLL_gradients_leader_deconv)
			#d_NLL_leader_deconv.append(NLL_gradients_tensor_leader_deconv)

			d_NLL_leader.append(torch.cat((NLL_gradients_tensor_leader_fc,NLL_gradients_tensor_leader_deconv)))
			#d_NLL_leader.append(d_NLL_leader_fc.cat(d_NLL_leader_deconv))

		d_NLL_chaser = torch.stack(d_NLL_chaser)
		d_NLL_leader = torch.stack(d_NLL_leader)
		kernel_matrix, grad_kernel, _ = self.get_kernel(particle_tensor=self.theta_am)

		q_chaser = self.theta_am - self.inner_lr*(torch.matmul(kernel_matrix, d_NLL_chaser) - grad_kernel)
		q_leader = self.theta_am - self.inner_lr*(torch.matmul(kernel_matrix, d_NLL_leader) - grad_kernel)

		#HASN'T BEEN CHANGED YET
		for _ in range(self.num_inner_updates):
			d_NLL_chaser = []
			d_NLL_leader = []
			for particle_id in range(self.num_particles):
				w = get_weights_target_net(w_generated=q_chaser, row_id=particle_id, w_target_shape=self.w_shape)
				#chaser
				y_pred_t = self.net.forward(x=x_t, w=w, p_dropout=self.p_dropout_base)
				loss_NLL_chaser = self.loss(y_pred_t, y_t)
		
				NLL_grads_chaser = torch.autograd.grad(
					outputs=loss_NLL_chaser,
					inputs=w.values(),
					create_graph=True
				)
				NLL_gradients_chaser = dict(zip(w.keys(), NLL_grads_chaser))
				NLL_gradients_tensor_chaser = self.dict2tensor(dict_obj=NLL_gradients_chaser)
				d_NLL_chaser.append(NLL_gradients_tensor_chaser)

				#leader
				x = torch.cat((x_t,x_v),0)
				w = get_weights_target_net(w_generated=q_leader, row_id=particle_id, w_target_shape=self.w_shape)
				y_pred = self.net.forward(x=x, w=w, p_dropout=self.p_dropout_base)
				loss_NLL_leader = self.loss(y_pred, (torch.cat((y_t,y_v),0)))

				NLL_grads_leader = torch.autograd.grad(
					outputs=loss_NLL_leader,
					inputs=w.values(),
					create_graph=True
				)
				NLL_gradients_leader = dict(zip(w.keys(), NLL_grads_leader))
				NLL_gradients_tensor_leader = self.dict2tensor(dict_obj=NLL_gradients_leader)
				d_NLL_leader.append(NLL_gradients_tensor_leader)
			d_NLL_chaser = torch.stack(d_NLL_chaser)
			d_NLL_leader = torch.stack(d_NLL_leader)
			kernel_matrix, grad_kernel, _ = self.get_kernel(particle_tensor=self.theta)

			q_chaser = q_chaser - self.inner_lr*(torch.matmul(kernel_matrix, d_NLL_chaser) - grad_kernel)
			q_leader = q_leader - self.inner_lr*(torch.matmul(kernel_matrix, d_NLL_leader) - grad_kernel)

		#remove this later
		#y_pred = []
		if predict is True:
			y_pred_v = []
			for particle_id in range(self.num_particles):
				w = get_weights_target_net(w_generated=q_chaser, row_id=particle_id, w_target_shape=self.w_shape)
				y_pred_ = self.net.forward(x=x_v, w=w, p_dropout=0)
				y_pred_v.append(y_pred_)
			return q_chaser, q_leader, y_pred_v
		
		#maybe return loss_NLL_leader as well
		return q_chaser, q_leader, loss_NLL_chaser

	
	def get_meta_loss(self, chaser, leader):
		loss = torch.dist(chaser, leader)
		return loss

	def get_kernel(self, particle_tensor):
		'''
		Compute the RBF kernel for the input particles
		Input: particles = tensor of shape (N, M)
		Output: kernel_matrix = tensor of shape (N, N)
		'''
		pairwise_d_matrix = self.get_pairwise_distance_matrix(particle_tensor)

		median_dist = torch.median(pairwise_d_matrix)  # tf.reduce_mean(euclidean_dists) ** 2
		h = median_dist / np.log(self.num_particles)

		kernel_matrix = torch.exp(-pairwise_d_matrix / h)
		kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
		grad_kernel = -torch.matmul(kernel_matrix, particle_tensor)
		grad_kernel += particle_tensor * kernel_sum
		grad_kernel /= h
		return kernel_matrix, grad_kernel, h

	def get_pairwise_distance_matrix(self, particle_tensor):
		'''
		Input: tensors of particles
		Output: matrix of pairwise distances
		'''
		num_particles = particle_tensor.shape[0]
		euclidean_dists = torch.nn.functional.pdist(input=particle_tensor, p=2) # shape of (N)

		# initialize matrix of pairwise distances as a N x N matrix
		pairwise_d_matrix = torch.zeros((num_particles, num_particles), device=self.device)

		# assign upper-triangle part
		triu_indices = torch.triu_indices(row=num_particles, col=num_particles, offset=1)
		pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

		# assign lower-triangle part
		#SHOULD WE NOT TRANSPOSE BACK??
		pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
		pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

		return pairwise_d_matrix

	def dict2tensor(self,dict_obj):
		d2tensor = []
		for key in dict_obj.keys():
			tensor_temp = torch.flatten(dict_obj[key], start_dim=0, end_dim=-1)
			d2tensor.append(tensor_temp)
		d2tensor = torch.cat(d2tensor)
		return d2tensor

	def predict(self, saved_checkpoint):
		checkpoint = torch.load(saved_checkpoint, map_location=self.device)
		theta_enc = checkpoint['theta_enc']
		theta_am = checkpoint['theta_am']

		train_data_loader = Data('./data', 'train')
		train_data = train_data_loader.get_data()
		keys = train_data_loader.keys
		scaler = Scaler(train_data[keys[0]])
		padded_data = scaler.pad_scale()
		x_t, y_t, x_v, y_v = train_data_loader.get_train_test(padded_data, device=self.device)

		y_pred = []
		for particle_id in range(self.num_particles):
			w_enc = get_weights_target_net(w_generated=theta_enc, row_id=particle_id, w_target_shape=self.w_shape)
			#chaser

			#extract features
			in_features = self.net.forward(x=x_t, w=w_enc)
			train_length = len(in_features)
			out_features = self.net.forward(x=y_t[:train_length-1], w=w_enc)
			
			#parameters
			w_fc = get_weights_target_net(w_generated=theta_am, row_id=particle_id, w_target_shape=self.w_shape_fc)
			task_params = task_inference(in_features[:train_length-1], out_features, weights=w_fc, device=self.device)
			
			#generate image
			sample_log_py = []
			task_vector = task_params['psi_samples'][0, :, :]
			#repeat this for the batch, CHECK IF REPEAT TENSOR IS THE RIGHT SIZE
			task_vector = task_vector.repeat(len(in_features[train_length-1:]), 1)
			task_vector = gen_input(task_vector, w_fc, in_features[train_length-1:])
			w_deconv = get_weights_target_net(w_generated=self.theta_am, row_id=particle_id, w_target_shape=self.w_shape_dec)
			y_pred_t = self.net.forward(x=task_vector, w=w_deconv, deconv=True, p_dropout=self.p_dropout_base,)
			y_pred.append(y_pred_t)
		return y_pred
	

#model = Bmaml(device=torch.device('cuda:0'))
#model.init_theta()
#model.meta_train()