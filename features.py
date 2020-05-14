import copy
import numpy as np 

def scale_up1(data):
	data = copy.deepcopy(data)

	for traintest in data:
		for instance in data[traintest]:
			for in_out in instance:
				scale = 30 // max(len(instance[in_out][0]), len(instance[in_out]))
				instance[in_out] = np.kron(np.array(instance[in_out]), np.ones((scale,scale))).tolist()

	return data

def scale_up2(data):
	data = copy.deepcopy(data)

	for traintest in data:
		for instance in data[traintest]:
			scale = 30 // max(len(instance['input'][0]), len(instance['input']), len(instance['output'][0]), len(instance['output']))
			instance['input'] = np.kron(np.array(instance['input']), np.ones((scale, scale))).tolist()
			if len(list(instance.keys())) == 2:
				instance['output'] = np.kron(np.array(instance['output']), np.ones((scale, scale))).tolist()

	return data