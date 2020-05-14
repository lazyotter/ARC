import torch
import torchvision

import numpy as np 
import random
import itertools
from scipy.special import erf

from sine_data_utils import get_num_weights, get_weights_target_net, get_task_sine_data
from DataGeneratorT import DataGenerator
from FCNet import FCNet
from load_data import Data

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
				 num_epochs=1000,
				 num_particles=10,
				 lr_decay=1,
				 meta_l2_regularization=0,
				 num_val_tasks=100,
				 p_dropout_base=0):

		self.num_training_samples_per_class = k
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
		self.net = FCNet()
		self.p_sine = 0.5
		#Chane this loss to chaser loss
		self.loss = torch.nn.MSELoss()
		self.theta = []
		self.op_theta = None
		self.w_shape = None
		self.num_epochs_save = 1
		#THIS SEEMS LIKE ITS JUST EXPECTED TOTAL TASKS PER EPOCH
		expected_tasks_save_loss = 2000
		self.num_tasks_save_loss = int(expected_tasks_save_loss/self.num_tasks_per_minibatch)*self.num_tasks_per_minibatch
		dst_folder_root = './output'
		self.dst_folder = '{0:s}/BMAML_few_shot/BMAML_{1:s}_{2:d}way_{3:d}shot'.format(
			dst_folder_root,
			'sine_line',
			self.num_classes_per_task,
			self.num_training_samples_per_class
		)
		self.device = torch.device('cuda:0')
		if not os.path.exists(self.dst_folder):
			os.makedirs(self.dst_folder)
			print('No folder for storage found')
			print('Make folder to store meta-parameters at')
		else:
			print('Found existing folder. Meta-parameters will be stored at')
		print(self.dst_folder)

	def init_theta(self):
		self.w_shape = self.net.get_weight_shape()
		num_weights = get_num_weights(self.net)
		print('Number of parameters of base model = {0:d}'.format(num_weights))
		for _ in range(self.num_particles):
			theta_flatten = []
			for key in self.w_shape.keys():
				if isinstance(self.w_shape[key], tuple):
					theta_temp = torch.empty(self.w_shape[key], device = self.device)
					torch.nn.init.xavier_normal_(tensor=theta_temp)
				else:
					theta_temp = torch.zeros(self.w_shape[key], device = self.device)
				theta_flatten.append(torch.flatten(theta_temp, start_dim=0, end_dim=-1))
			self.theta.append(torch.cat(theta_flatten))
		self.theta = torch.stack(self.theta)
		self.theta.requires_grad_()
		self.op_theta = torch.optim.Adam(
			params=[self.theta],
			lr = self.meta_lr)

	def meta_train(self, train_subset='train'):
		if train_subset == 'train':
			train_data_loader = Data('./data', 'train')
			train_data = train_data_loader.get_data()
		keys = train_data_loader.keys


		print('Start to train...')
		for epoch in range(0, self.num_epochs):
			#variables for monitoring
			meta_loss_saved = []
			val_accuracies = []
			train_accuracies = []

			meta_loss = 0 #accumulate the loss of many ensembling networks
			num_meta_updates_count = 0

			meta_loss_avg_print = 0
			#meta_mse_avg_print = 0

			meta_loss_avg_save = []
			#meta_mse_avg_save = []

			task_count = 0

			#change to maybe do multiple tasks
			for key in keys:
				padded_data = train_data_loader.pad(train_data[key])
				x_t, y_t, x_v, y_v = train_data_loader.get_train_test(padded_data)

				chaser, leader, y_pred = self.get_task_prediction(x_t, y_t, x_v, y_v)
				loss_NLL = self.get_meta_loss(chaser, leader)

				if torch.isnan(loss_NLL).item():
					sys.exit('NaN error')

				meta_loss = meta_loss + loss_NLL
				#meta_mse = self.loss(y_pred, y_v)

				task_count = task_count + 1

				if task_count % self.num_tasks_per_minibatch == 0:
					meta_loss = meta_loss/self.num_tasks_per_minibatch
					#meta_mse = meta_mse/self.num_tasks_per_minibatch

					# accumulate into different variables for printing purpose
					meta_loss_avg_print += meta_loss.item()
					#meta_mse_avg_print += meta_mse.item()

					self.op_theta.zero_grad()
					meta_loss.backward()
					self.op_theta.step()

					# Printing losses
					num_meta_updates_count += 1
					if (num_meta_updates_count % self.num_meta_updates_print == 0):
						meta_loss_avg_save.append(meta_loss_avg_print/num_meta_updates_count)
						#meta_mse_avg_save.append(meta_mse_avg_print/num_meta_updates_count)
						print('{0:d}, {1:2.4f}, {1:2.4f}'.format(
							task_count,
							meta_loss_avg_save[-1]
							#meta_mse_avg_save[-1]
						))

						num_meta_updates_count = 0
						meta_loss_avg_print = 0
						#meta_mse_avg_print = 0
					
					if (task_count % self.num_tasks_save_loss == 0):
						meta_loss_saved.append(np.mean(meta_loss_avg_save))

						meta_loss_avg_save = []
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

				if (task_count >= self.num_tasks_per_epoch):
					break
			if ((epoch + 1)% self.num_epochs_save == 0):
				checkpoint = {
					'theta': self.theta,
					'meta_loss': meta_loss_saved,
					'val_accuracy': val_accuracies,
					'train_accuracy': train_accuracies,
					'op_theta': self.op_theta.state_dict()
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
			w = get_weights_target_net(w_generated=self.theta, row_id=particle_id, w_target_shape=self.w_shape)
			#chaser
			#extract features
			#parameters
			#generate image
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

		q_chaser = self.theta - self.inner_lr*(torch.matmul(kernel_matrix, d_NLL_chaser) - grad_kernel)
		q_leader = self.theta - self.inner_lr*(torch.matmul(kernel_matrix, d_NLL_leader) - grad_kernel)

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
		y_pred = []
		if predict is True:
			y_pred_v = []
			for particle_id in range(self.num_particles):
				w = get_weights_target_net(w_generated=q_chaser, row_id=particle_id, w_target_shape=self.w_shape)
				y_pred_ = self.net.forward(x=x_v, w=w, p_dropout=0)
				y_pred_v.append(y_pred_)
			return q_chaser, q_leader, y_pred_v
		
		return q_chaser, q_leader, y_pred

	
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

	def meta_validation(self, datasubset, num_val_tasks, return_uncertainty=False):
		x0 = torch.linspace(start=-5, end=5, steps=100, device=self.device).view(-1, 1) # vector

		quantiles = np.arange(start=0., stop=1.1, step=0.1)
		cal_data = []

		data_generator = DataGenerator(num_samples=self.num_training_samples_per_class, device=self.device)
		for _ in range(num_val_tasks):

			# generate sinusoidal data
			x_t, y_t, amp, phase, slope = data_generator.generate_sinusoidal_data(noise_flag=True)
			y0 = amp*torch.sin(slope*x0 + phase)
			y0 = y0.view(1, -1).cpu().numpy() # row vector
			
			y_preds = torch.stack(self.get_task_prediction(x_t=x_t, y_t=y_t, x_v=x0)) # K x len(x0)

			y_preds_np = torch.squeeze(y_preds, dim=-1).detach().cpu().numpy()
			
			y_preds_quantile = np.quantile(a=y_preds_np, q=quantiles, axis=0, keepdims=False)

			# ground truth cdf
			std = data_generator.noise_std
			cal_temp = (1 + erf((y_preds_quantile - y0)/(np.sqrt(2)*std)))/2
			cal_temp_avg = np.mean(a=cal_temp, axis=1) # average for a task
			cal_data.append(cal_temp_avg)
		return cal_data

model = Bmaml()
model.init_theta()
model.meta_train()