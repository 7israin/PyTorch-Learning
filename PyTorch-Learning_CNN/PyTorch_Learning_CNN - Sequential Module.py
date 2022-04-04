# Firstly, we handle our imports.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as udata

import torchvision
import torchvision.transforms as transforms
#PyTorch imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix	#from plotcm import plot_confusion_matrix
#data science in Python

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import itertools
from itertools import product
from collections import OrderedDict
from collections import namedtuple
from IPython.display import display, clear_output

import pdb  #Python debugger
import time
import json
from datetime import datetime

torch.set_printoptions(linewidth = 120)

EPOCH = 5		   # 训练整批数据多少次	
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False
SAVE_MODEL = False # 保存本次训练好的模型
LOAD_MODEL = False # 载入以前训练好的模型
MODEL_NUM = 54168 # 本模型特征码
# Hyper Parameters
params = OrderedDict(
	p_lr = [.001] # 学习率
	,p_batch_size = [1000]
	,p_shuffle = [False]
	,num_workers = [2]
    ,device = ['cuda']
)

# Then, create a normalized dataset
train_set = torchvision.datasets.FashionMNIST(
	root = './data'
	,train = True
	,download = DOWNLOAD_MNIST
	,transform = transforms.Compose([
		transforms.ToTensor()
	])
)

class CaculNormal():
	def __init__(self, train_set):
		# create a data loader with a smaller batch size
		loader = DataLoader(train_set, batch_size=1000, num_workers=0)

		# calculate our n value or total number of pixels
		num_of_pixels = len(train_set) * 28 * 28 # is the height and width of the images inside our dataset

		# sum the pixels values by iterating over each batch
		total_sum = 0
		for batch in loader: total_sum += batch[0].sum()
		self.mean = total_sum / num_of_pixels

		# calculate the sum of the squared errors
		sum_of_squared_error = 0
		for batch in loader: 
			sum_of_squared_error += ((batch[0] - self.mean).pow(2)).sum()
		self.std = torch.sqrt(sum_of_squared_error / num_of_pixels)

cn = CaculNormal(train_set)

train_set_normal = torchvision.datasets.FashionMNIST(
    root = './data'
    ,train = True
    ,download = DOWNLOAD_MNIST
    ,transform = transforms.Compose([
          transforms.ToTensor()
        , transforms.Normalize(cn.mean, cn.std)
    ])
)


class RunBuilder():
	@staticmethod
	def get_runs(params):

		Run = namedtuple('Run', params.keys())

		runs = []
		for v in product(*params.values()):
			runs.append(Run(*v))

		return runs

class RunManager():
	def __init__(self):
	
		self.epoch_count = 0
		self.epoch_loss = 0
		self.epoch_num_correct = 0
		self.epoch_start_time = None
	
		self.run_params = None
		self.run_count = 0
		self.run_data = []
		self.run_start_time = None
	
		self.network = None
		self.loader = None
		self.tb = None

	def begin_run(self, run, network, loader):

		self.run_start_time = time.time() # capture the start time for the run

		self.run_params = run # save the passed in run parameters
		self.run_count += 1 # increment the run count by one

		self.network = network # save our network
		self.loader = loader # save our data loader
		self.tb = SummaryWriter(comment=f'-{run}') # initialize a SummaryWriter for TensorBoard

		images, labels = next(iter(self.loader))
		grid = torchvision.utils.make_grid(images)

		self.tb.add_image('images', grid)
		self.tb.add_graph(
				self.network
				, images.to(getattr(run, 'device', 'cpu'))
		)

	def end_run(self):
		self.tb.close()
		self.epoch_count = 0

	def begin_epoch(self):
		self.epoch_start_time = time.time()

		self.epoch_count += 1
		self.epoch_loss = 0
		self.epoch_num_correct = 0

	def end_epoch(self):

		epoch_duration = time.time() - self.epoch_start_time # the epoch duration is final
		run_duration = time.time() - self.run_start_time # the running time of the current run

		loss = self.epoch_loss / len(self.loader.dataset) # relative to the size of the training set
		accuracy = self.epoch_num_correct / len(self.loader.dataset)

		self.tb.add_scalar('Loss', loss, self.epoch_count)
		self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
		
		# pass our network's weights and gradient values to TensorBoard like we did before
		for name, param in self.network.named_parameters():
			self.tb.add_histogram(name, param, self.epoch_count)
			self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

		# a dictionary that contains the keys and values we care about for our run
		results = OrderedDict()
		results["run"] = self.run_count
		results["epoch"] = self.epoch_count
		results['loss'] = loss
		results["accuracy"] = accuracy
		results['epoch duration'] = epoch_duration
		results['run duration'] = run_duration

		""" iterate over the keys and values inside our run parameters adding them to the results dictionary.
		This will allow us to see the parameters that are associated with the performance results. """
		for k,v in self.run_params._asdict().items(): results[k] = v

		self.run_data.append(results) # append the results to the run_data list

		# turn the data list into a pandas data frame so we can have formatted output
		df = pd.DataFrame.from_dict(self.run_data, orient='columns')

	def track_loss(self, loss, batch):
		self.epoch_loss += loss.item() * batch[0].shape[0]

	def track_num_correct(self, preds, labels):
		self.epoch_num_correct += self._get_num_correct(preds, labels)

	def _get_num_correct(self, preds, labels):
		return preds.argmax(dim=1).eq(labels).sum().item()

	def save(self, fileName):

		pd.DataFrame.from_dict(
			self.run_data, orient='columns'
		).to_csv(f'{fileName}.csv')

		with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
			json.dump(self.run_data, f, ensure_ascii=False, indent=4)

network = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.BatchNorm2d(6)

    , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Flatten(start_dim=1)

    , nn.Linear(in_features=12*4*4, out_features=120)
    , nn.ReLU()
    , nn.BatchNorm1d(120)

    , nn.Linear(in_features=120, out_features=60)
    , nn.ReLU()

    , nn.Linear(in_features=60, out_features=10) # out_classes = len(train_set.classes) = 10 
)

if LOAD_MODEL:
    network.load_state_dict(torch.load("./network_" + MODEL_NUM + ".pt"))

torch.set_grad_enabled(True)
rm = RunManager()

def process():
	for run in RunBuilder.get_runs(params):
		comment = f'-{run}'

		device = torch.device(run.device)
		network.to(device)

		train_loader = DataLoader(
			train_set_normal
			,batch_size=run.p_batch_size
			,shuffle=run.p_shuffle
			,num_workers=run.num_workers
		)

		optimizer = optim.Adam(
			network.parameters(), lr=run.p_lr
		)

		rm.begin_run(run, network, train_loader)

		for epoch in range(EPOCH):
			rm.begin_epoch()
			for batch in train_loader: # Get Batch
				images = batch[0].to(device)
				labels = batch[1].to(device)

				preds = network(images) # Pass Batch
				loss = F.cross_entropy(preds, labels) # Calculate Loss

				optimizer.zero_grad() # Zero Gradients
				loss.backward() # Calculate Gradients
				optimizer.step() # Update Weights

				rm.track_loss(loss, batch)
				rm.track_num_correct(preds, labels)

			rm.end_epoch()
		
			print(
				"epoch", rm.epoch_count, 
				"total_correct:", rm.epoch_num_correct, 
				"loss:", rm.epoch_loss
			)

		rm.end_run()
	rm.save('results_' + datetime.now().strftime('%b.%d_%H-%M-%S'))

if __name__ == '__main__':
    process()

if SAVE_MODEL:
    torch.save(network.state_dict(), "./network_" + str(MODEL_NUM) + ".pt")