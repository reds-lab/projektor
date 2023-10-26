import otdd
from otdd.pytorch.datasets import load_imagenet, load_torchvision_data, load_torchvision_data_shuffle, load_torchvision_data_perturb, load_torchvision_data_keepclean
from otdd.pytorch.distance import DatasetDistance, FeatureCost

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

import matplotlib.pyplot as plt
from torch import tensor
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from copy import deepcopy as dpcp
import pickle 
import time

# import torchshow as ts

from torchvision.utils import make_grid
from torch.utils.data import random_split, Dataset, TensorDataset, DataLoader

import argparse



parser = argparse.ArgumentParser()
# add_dataset_model_arguments(parser)

parser.add_argument('--cnum', type=int, required=True,
                    help='number of cuda in the server')

parser.add_argument('--n', type=int, required=True,
                    help='number of data')
arg = parser.parse_args() # args conflict with other argument

print(f"procs cnum {arg.cnum}")

print(f"data cnum {arg.n}")
    
print("end")


cuda_num = arg.cnum
import torch
print(torch.__version__)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_num)
print(os.environ["CUDA_VISIBLE_DEVICES"])
torch.cuda.set_device(cuda_num)
print("Cuda device: ", torch.cuda.current_device())
print("cude devices: ", torch.cuda.device_count())
device = torch.device('cuda:' + str(cuda_num) if torch.cuda.is_available() else 'cpu')


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
data_all = pickle.load( open('data/cifar10.data', 'rb') )
train_features, train_labels, test_features, test_labels  = data_all


# data_all = pickle.load(open('Baselines/datasets/clean_cifar.data', 'rb'))
# train_features, train_labels, test_features, test_labels  = data_all

label_idx = []
for i in range(10):
    label_idx.append((train_labels==i).nonzero()[0])
    
test_label_idx = []
for i in range(10):
    test_label_idx.append((test_labels==i).nonzero()[0])
    
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, 10)
#         self.linear1 = nn.Linear(128, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
#         return out # only for embedder
#         out = self.linear1(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])


def get_model_log_err(train_loader, test_loader, epochs = 110):
    
    net = PreActResNet18()
    net = net.to(device)

    test_criterion = nn.CrossEntropyLoss()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    best_train_loss = 999999
    for epoch in range(epochs):
        # Training
    #     print('Epoch {}/{}'.format(epoch + 1, 70))
    #     print('-' * 10)
        start_time = time.time()
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        end_time = time.time()
        if epoch % 10 == 0:
            print('%.1f . TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec ' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))
        best_train_loss = min(best_train_loss, train_loss/(batch_idx+1))
        
            
#         net.eval()
        test_loss = 0
        correct = 0
        total = 0
#         acc = [0 for c in list_of_classes]

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = test_criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # class wise accuracy
            
    
        if epoch % 10 == 0:
            print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
    test_loss /= (batch_idx+1)    
    print(f"test loss {test_loss} train loss {best_train_loss}")
        
    return test_loss - best_train_loss, 100.*correct/total

def get_ot_dist(train_loader, test_loader, n=5000):
    
    
    net_test = PreActResNet18()
    net_test = net_test.to(device)
    net_test.load_state_dict(torch.load('checkpoint/preact_resnet18.pth', map_location=str('cuda:'+str(cuda_num))))
    net_test.eval()

    embedder = net_test.to(device)
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding = embedder,
                               src_dim = (3,32,32),
                               tgt_embedding = embedder,
                               tgt_dim = (3,32,32),
                               p = 2,
                               device='cuda')

    dist = DatasetDistance(train_loader, test_loader,
                           inner_ot_method = 'exact',
                           debiased_loss = True,
                           feature_cost = feature_cost,
                           λ_x=1.0, λ_y=1.0,
                           sqrt_method = 'spectral',
                           sqrt_niters=10,
                           precision='single',
                           p = 2, entreg = 1e-2,
                           device='cuda')
    k = dist.distance(maxsamples = n, return_coupling = True)

    return k[0].item()

def dataset_q(q1_amt, q2_amt, num, train_feats, train_labels):
    # two datasets, q=0 -> dataset2, q=1 -> dataset1
    # validation set: unbiased sample from MNIST validation set
    # dataset1: class 0-4: 99% (19.8% each class), class 5-9: 1% (0.2% each class)
    # dataset2: class 0-4: 2% (0.4% each class), class 5-9: 98% (19.6% each class)
    # near balance at q=0.5

    ds1_idx = []
    ds2_idx = []
    ds3_idx = []
    ds1_labels = []
    ds2_labels = []
    ds3_labels = []
    # ds1_features = []
    # ds2_features = []

    d1c1 = 0.2425
    d1c2 = 0.005
    d1c3 = 0.005

    d2c1 = 0.0057
    d2c2 = 0.32
    d2c3 = 0.0057

    d3c1 = 0.0014
    d3c2 = 0.0014
    d3c3 = 0.33
    
    
    
    # sample size
    n = num # size of dataset for training (use for construct)
    # ratio
    q1 = q1_amt # q * dataset 1
    q2 = q2_amt # q * dataset 1
    q3 = 1-q1-q2 # q * dataset 1

    for i in range(4):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c1)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c1)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c1)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c1)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c1)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c1)))*i)
    for i in range(4, 7):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c2)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c2)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c2)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c2)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c2)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c2)))*i)
    for i in range(7, 10):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c3)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c3)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c3)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c3)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c3)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c3)))*i)

    ds1_features_fl = train_feats[np.concatenate(ds1_idx)]
    ds2_features_fl = train_feats[np.concatenate(ds2_idx)]
    ds3_features_fl = train_feats[np.concatenate(ds3_idx)]
    ds1_features = train_feats[np.concatenate(ds1_idx)]
    ds2_features = train_feats[np.concatenate(ds2_idx)]
    ds3_features = train_feats[np.concatenate(ds3_idx)]
    train_x_2d = np.concatenate([ds1_features, ds2_features, ds3_features])

    ds1_labels = np.concatenate(ds1_labels)
    ds2_labels = np.concatenate(ds2_labels)
    ds3_labels = np.concatenate(ds3_labels)

    train_x = np.concatenate([ds1_features_fl, ds2_features_fl, ds3_features_fl])
    train_y = np.concatenate([ds1_labels, ds2_labels, ds3_labels])
            
    
    return train_x, train_y


n = arg.n
q = 0.0
batch_size = 256

breaks = 10
reps = 3

# make test dataloader

test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_features).permute(0,3,1,2), torch.LongTensor(test_labels)), 
                                                batch_size=batch_size, 
                                                shuffle=False)

qsreserrlog = []
qsotlog = []
qsaccs = []
for l in range(breaks+1):
    reserrlog = []
    otlog = []
    accs = []
    
    for j in range(breaks+1): # going through q, from 0 to 1 - 20 points
        start_t = time.time()
        q1 = l/10
        q2 = j/10
        q3 = 1-q1-q2
        if q3<0:
            break

        cacheerr = []
        cacheot = []
        cacheacc = []

        # create dataset
        train_x, train_y = dataset_q(q1, q2, n, train_features, train_labels)

        # make train dataloader
        train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_x).permute(0,3,1,2), 
                                            torch.LongTensor(train_y)), 
                                            batch_size=batch_size, 
                                            shuffle=True)
        for rep in range(reps):
            # get OT dist
            cacheot.append(get_ot_dist(train_loader, test_loader, n=n))
            loss, acc = get_model_log_err(train_loader, test_loader)
            # get model error (test loss - train loss)
            cacheacc.append(acc)
            cacheerr.append(loss)
        print("cacheerr: ", cacheerr)
        print("cacheot: ", cacheot)
        print("cacheacc: ", cacheacc)
        # add median of vals
        reserrlog.append(np.median(cacheerr)) # median then loss + no need for log
        otlog.append(np.median(cacheot))
        accs.append(np.median(cacheacc))
    print("j: ", j, " it took: ", time.time() - start_t)
    
    qsreserrlog.append(reserrlog)
    qsotlog.append(otlog)
    qsaccs.append(accs)
pickle.dump([qsreserrlog,qsotlog,qsaccs], open(f'projektor_data/cif10_3sources_unbalanced_{n}_br_{breaks}_rep_{reps}.res', 'wb' ))
