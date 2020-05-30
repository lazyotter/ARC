import torch
from utilities import dense_layer, sample_normal

def task_inference(in_features, out_features, weights, device, num_samples=1):
	h = torch.cat((in_features, out_features), -1)

	#dense layer before pooling
	h = dense_layer(h, weights, layer = 1)
	h = dense_layer(h, weights, layer = 2)

	#pool across dimensions
	nu = torch.unsqueeze(torch.mean(h, dim=0), dim=0)
	post_processed = _post_process(nu, weights)

	#compute mean and log variance of each parameter
	psi = {}
	psi['mu'] = dense_layer(post_processed, weights, layer = 5, activation = None)
	psi['log_variance'] = dense_layer(post_processed, weights, layer = 6, activation = None)
	
	psi['psi_samples'] = sample_normal(psi['mu'], psi['log_variance'], num_samples, device=device)
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