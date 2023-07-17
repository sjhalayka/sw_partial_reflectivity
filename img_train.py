import numpy as np
import math
import cv2
import random
import torch
from torch import flatten
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import os.path
from os import path

import time




img_width = 32
num_channels = 3

#num_input_components = img_width*img_width*num_channels
num_output_components = 2

num_epochs = 10
learning_rate = 0.001

max_train_files = 100000

num_recursions = 10
num_child_networks = 5




class Net(torch.nn.Module):

	def __init__(self, num_channels, num_output_components):

		super().__init__()
		self.model = torch.nn.Sequential(
		    #Input = 3 x 32 x 32, Output = 32 x 32 x 32
		    torch.nn.Conv2d(in_channels = num_channels, out_channels = 32, kernel_size = 3, padding = 1), 
		    torch.nn.ReLU(),
		    #Input = 32 x 32 x 32, Output = 32 x 16 x 16
		    torch.nn.MaxPool2d(kernel_size=2),
  
		    #Input = 32 x 16 x 16, Output = 64 x 16 x 16
		    torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
		    torch.nn.ReLU(),
		    #Input = 64 x 16 x 16, Output = 64 x 8 x 8
		    torch.nn.MaxPool2d(kernel_size=2),
		      
		    #Input = 64 x 8 x 8, Output = 64 x 8 x 8
		    torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
		    torch.nn.ReLU(),
		    #Input = 64 x 8 x 8, Output = 64 x 4 x 4
		    torch.nn.MaxPool2d(kernel_size=2),
  
		    torch.nn.Flatten(),
		    torch.nn.Linear(1024, 256),
		    torch.nn.ReLU(),
		    torch.nn.Linear(256, num_output_components)
		)
  
	def forward(self, x):
		return self.model(x)








class float_image:

	def __init__(self, img):
		self.img = img

class image_type:

	def __init__(self, img_type, float_img):
		self.img_type = img_type
		self.float_img = float_img





def do_network(in_net, batch, ground_truth, num_channels, num_output_components, all_train_files, random_seed, num_epochs):

	if (in_net is None):
		in_net = Net(num_channels, num_output_components)

	net = in_net

	random.seed(random_seed)

	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
	loss_func = torch.nn.MSELoss()

	loss = 0;

	for epoch in range(num_epochs):
		
		random.shuffle(all_train_files)

		count = 0

		for i in all_train_files:

			batch[count] = i.float_img
		
			if i.img_type == 0: # cat

				ground_truth[count][0] = 1
				ground_truth[count][1] = 0
			
			elif i.img_type == 1: # dog
				
				ground_truth[count][0] = 0
				ground_truth[count][1] = 1

			count = count + 1
	
		x = Variable(torch.from_numpy(batch))
		y = Variable(torch.from_numpy(ground_truth))

		prediction = net(x)	 
		loss = loss_func(prediction, y)

		print(epoch, loss)

		optimizer.zero_grad()	 # clear gradients for next train
		loss.backward()		 # backpropagation, compute gradients
		optimizer.step()		# apply gradients
	
	return net, loss






if False: #path.exists('weights_' + str(img_width) + '_' + str(num_epochs) + '.pth'):
	net.load_state_dict(torch.load('weights_' + str(img_width) + '_' + str(num_epochs) + '.pth'))
	print("loaded file successfully")
else:
	print("training...")





	all_train_files = []




	file_count = 0

	path = 'training_set/cats/'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		file_count = file_count + 1
		if file_count >= max_train_files:
			break;

		print(path + f)
		img = cv2.imread(path + f)
		
		if (img is None) == False:

			img = img.astype(np.float32)
			res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
			flat_file = res / 255.0
			flat_file = np.transpose(flat_file, (2, 0, 1))
			all_train_files.append(image_type(0, flat_file))

		else:
			print("image read failure")





	file_count = 0

	path = 'training_set/dogs/'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		file_count = file_count + 1
		if file_count >= max_train_files:
			break;

		print(path + f)
		img = cv2.imread(path + f)
		
		if (img is None) == False:

			img = img.astype(np.float32)
			res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
			flat_file = res / 255.0
			flat_file = np.transpose(flat_file, (2, 0, 1))
			all_train_files.append(image_type(1, flat_file))

		else:
			print("image read failure")




	batch = np.zeros((len(all_train_files), num_channels, img_width, img_width), dtype=np.float32)
	ground_truth = np.zeros((len(all_train_files), num_output_components), dtype=np.float32)	

	curr_net, curr_loss = do_network(None, batch, ground_truth, num_channels, num_output_components, all_train_files, round(time.time()), num_epochs)

	for y in range(num_recursions):
		for x in range(num_child_networks):

			print(y, x)

			net, loss = do_network(curr_net, batch, ground_truth, num_channels, num_output_components, all_train_files, round(time.time()), num_epochs)

			if loss < curr_loss:

				curr_loss = loss
				curr_net = net







#	torch.save(net.state_dict(), 'weights_' + str(img_width) + '_' + str(num_epochs) + '.pth')



path = 'test_set/cats/'
filenames = next(os.walk(path))[2]

cat_count = 0
total_count = 0

for f in filenames:

	img = cv2.imread(path + f)
			
	if (img is None) == False:

		img = img.astype(np.float32)
		res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
		flat_file = res / 255.0
		flat_file = np.transpose(flat_file, (2, 0, 1))

	else:

		print("image read failure")
		continue

	batch = torch.zeros((1, num_channels, img_width, img_width), dtype=torch.float32)
	batch[0] = torch.from_numpy(flat_file)

	prediction = curr_net(Variable(batch))

	if prediction[0][0] > prediction[0][1]:
		cat_count = cat_count + 1

	total_count = total_count + 1

print(cat_count / total_count)
print(total_count)





path = 'test_set/dogs/'
filenames = next(os.walk(path))[2]

dog_count = 0
total_count = 0

for f in filenames:

	img = cv2.imread(path + f)
			
	if (img is None) == False:

		img = img.astype(np.float32)
		res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
		flat_file = res / 255.0
		flat_file = np.transpose(flat_file, (2, 0, 1))

	else:

		print("image read failure")
		continue

	batch = torch.zeros((1, num_channels, img_width, img_width), dtype=torch.float32)
	batch[0] = torch.from_numpy(flat_file)

	prediction = curr_net(Variable(batch))

	if prediction[0][0] < prediction[0][1]:
		dog_count = dog_count + 1

	total_count = total_count + 1

print(dog_count / total_count)
print(total_count)