import colorama
import torch
import pdb
import traceback
from colorama import Fore, Back, Style
from torch import autograd

colorama.init()

class GuruMeditation (autograd.detect_anomaly):  
	
	def __init__(self):
		super(GuruMeditation, self).__init__()  

	def __enter__(self):
		super(GuruMeditation, self).__enter__()
		return self  

	def __exit__(self, type, value, trace):
		super(GuruMeditation, self).__exit__()
		if isinstance(value, RuntimeError):
			traceback.print_tb(trace)
			halt(str(value))

def halt(msg):
	print (Fore.RED + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
	print (Fore.RED + "┃ Software Failure. Press left mouse button to continue ┃")
	print (Fore.RED + "┃        Guru Meditation 00000004, 0000AAC0             ┃")
	print (Fore.RED + "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
	print(Style.RESET_ALL)
	print (msg)
	pdb.set_trace()


def printer(element, name):

	if torch.is_tensor(element):
		torch_sum, nan, inf = print_tensor(element)
		if nan != 0 or inf != 0:
			print(f'{name} is a tensor')
			print('sum: ', torch_sum)
			print('number of nans: ', nan)
			print('number of infs: ', inf)
			print('shape: ', element.size())

	elif type(element) == list:
		total_sum, total_nan, total_inf = print_list(element)
		if total_nan != 0 or total_inf != 0:
			print(f'{name} is a list')
			print(f'sum: {total_sum}')
			print(f'number of nans: {total_nan}')
			print(f'number of infs: {total_inf}')

	elif type(element) == dict:
		dict_sum = 0
		dict_nan = 0
		dict_inf = 0
		keys = list(element.keys())
		for key in keys:
			if torch.is_tensor(element[key]):
				torch_sum, nan, inf = print_tensor(element[key])
				dict_sum += torch_sum
				dict_nan += nan
				dict_inf += inf

			elif type(element[key]) == list:
				list_sum, nan, inf = print_list(element[key])
				dict_sum += list_sum
				dict_nan += nan
				dict_inf += inf
			else:
				print('dict is not made up of tensors or list')
		if dict_nan != 0 or dict_inf != 0:
			print(f'{name} is a dict')
			print(f'sum: {dict_sum}')
			print(f'number of nans: {dict_nan}')
			print(f'number of infs: {dict_inf}')

	#else:
	#	print(f'{name} is not a tensor, list, or dict')

	#print()
	#print()


def print_list(element):
	total_sum = 0
	total_nan = 0
	total_inf = 0
	for tensor in element:
		if torch.is_tensor(tensor):
			total_sum += torch.sum(tensor)
			total_nan += torch.sum(torch.isnan(tensor))
			total_inf += torch.sum(torch.isinf(tensor))
		else:
			print('list does not contain tensors')

	return total_sum, total_nan, total_inf

def print_tensor(element):
	return torch.sum(element), torch.sum(torch.isnan(element)), torch.sum(torch.isinf(element))