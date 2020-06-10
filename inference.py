import torch
from utilities import dense_layer, sample_normal
from debugger import printer

def task_inference(in_features, angles, weights, device, num_samples=1):

	h = torch.cat((in_features, angles), -1)
	printer(h, 'cat')

	#dense layer before pooling
	h = dense_layer(h, weights, layer = 1)
	h = dense_layer(h, weights, layer = 2)
	printer(h, 'h')

	#pool across dimensions
	nu = torch.unsqueeze(torch.mean(h, dim=0), dim=0)
	printer(nu, 'mean')
	post_processed = _post_process(nu, weights)
	printer(post_processed, 'post_processed')
	#compute mean and log variance of each parameter
	psi = {}
	psi['mu'] = dense_layer(post_processed, weights, layer = 5, activation = None)
	printer(psi['mu'], 'mu')
	psi['log_variance'] = dense_layer(post_processed, weights, layer = 6, activation = None)
	printer(psi['log_variance'], 'log_variance')

	psi['psi_samples'] = sample_normal(psi['mu'], psi['log_variance'], num_samples, device=device)
	printer(psi['psi_samples'], 'psi_samples')
	return psi

def _post_process(pooled, weights):
	"""
	Process a pooled variable through 2 dense layers
	:param pooled: tensor of rank (1 x num_features).
	:param units: integer number of output features.
	:return: tensor of rank (1 x units)
	"""

	h = dense_layer(pooled, weights, layer = 3)
	h = dense_layer(h, weights, layer = 4)

	return h