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

from sklearn.metrics import confusion_matrix    #from plotcm import plot_confusion_matrix
#data science in Python

import itertools
import pdb  #Python debugger

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次
BATCH_SIZE = 100
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False

torch.set_printoptions(linewidth = 120)

train_set = torchvision.datasets.FashionMNIST(
    root = './data'
    ,train = True
    ,download = DOWNLOAD_MNIST
    ,transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

"""
train_loader = udata.DataLoader(
	train_set
    ,batch_size = 1000
    ,shuffle = True
)

print (len(train_set),"\n")
print (train_set.targets)
"""

"""
sample = next(iter(train_set))
image, label = sample

plt.imshow(image.squeeze(), cmap = "gray")
plt.show()
print (torch.tensor(label))
"""

"""
display_loader = udata.DataLoader(
    train_set, batch_size = 10
)

batch = next(iter(display_loader))
print('len:', len(batch))

images, labels = batch

grid = torchvision.utils.make_grid(images, nrow = 10)

plt.figure(figsize = (15,15))
plt.imshow(np.transpose(grid, (1,2,0)))
plt.show()
print('labels:', labels)
"""

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # implement the forward pass
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t

torch.set_grad_enabled(True)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

network = Network()
#print(network)

#if torch.cuda.is_available():
#    network.to('cuda')

"""
torch.set_grad_enabled(False)
sample = next(iter(train_set)) 
image, label = sample
pred = network(image.unsqueeze(0))
print(label)
print(pred.argmax(dim=1))
"""

train_loader = udata.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(network.parameters(), lr=LR)

"""
batch = next(iter(train_loader)) # Getting a batch
images, labels = batch

preds = network(images) # Pass Batch
loss = F.cross_entropy(preds, labels) # Calculating the loss

loss.backward() # Calculate Gradients
optimizer.step() # Updating the weights

print('loss1:', loss.item())
preds = network(images)
loss = F.cross_entropy(preds, labels)
print('loss2:', loss.item())
"""

"""
total_loss = 0
total_correct = 0

for batch in train_loader: # Get Batch
    images, labels = batch 

    preds = network(images) # Pass Batch
    loss = F.cross_entropy(preds, labels) # Calculate Loss

    optimizer.zero_grad()
    loss.backward() # Calculate Gradients
    optimizer.step() # Update Weights

    total_loss += loss.item()
    total_correct += get_num_correct(preds, labels)

print(
    "epoch:", 0, 
    "total_correct:", total_correct, 
    "loss:", total_loss
)
"""

for epoch in range(EPOCH):

    total_loss = 0
    total_correct = 0

    for batch in train_loader: # Get Batch
        images, labels = batch 

        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(
        "epoch", epoch, 
        "total_correct:", total_correct, 
        "loss:", total_loss
    )

@torch.no_grad() # Getting predictions while not training
def get_all_preds(model, loader):
    all_preds = torch.tensor([]) # Creates an empty tensor
    for batch in loader: # Iterates over the batches coming from the data loader
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat( # Concatenates the output predictions with the all_preds tensor
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

with torch.no_grad():
    prediction_loader = udata.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

# preds_correct = get_num_correct(train_preds, train_set.targets)

stacked = torch.stack( # Stack these two tensors along the second dimension
    (
        train_set.targets
        ,train_preds.argmax(dim=1)
    )
    ,dim=1
)

"""
cmt = torch.zeros(10,10, dtype=torch.int64) # Create the matrix with ten prediction categories
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
"""

cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))

plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, train_set.classes)
plt.show()