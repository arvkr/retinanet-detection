from google.colab import drive
drive.mount('/content/drive')
!ls /content/drive/My\ Drive/pytorch_hackathon

!git clone https://github.com/arvkr/retinanet-detection.git
cd retinanet-detection/

!pip3 install cffi
!pip3 install pandas
!pip3 install cython
!pip3 install opencv-python
!pip3 install requests

cd lib/nms
!rm -rf build/
# !rm *so
cd ../
!python setup3.py build_ext --inplace
cd ../

import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

# import coco_eval
import csv_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))
weights_path = '/content/drive/My Drive/pytorch_hackathon/coco_resnet_50_map_0_335_state_dict.pt'
state_dict = torch.load(weights_path)
state_dict.pop('classificationModel.output.bias')
state_dict.pop('classificationModel.output.weight')

class arguments:
  def __init__(self):
    self.dataset = 'csv'
    self.csv_train = './annotations.csv'
    self.csv_classes = './classes.csv'
    self.depth = 50
    self.epochs = 1
    self.csv_val = None
parser = arguments()

dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

if parser.csv_val is None:
  dataset_val = None
  print('No validation annotations provided.')

sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

if dataset_val is not None:
  sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
  dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
retinanet.load_state_dict(state_dict, strict = False)

use_gpu = True

if use_gpu:
  retinanet = retinanet.cuda()

retinanet = torch.nn.DataParallel(retinanet).cuda()

retinanet.training = True

optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

loss_hist = collections.deque(maxlen=500)

retinanet.train()
retinanet.module.freeze_bn()

print('Num training images: {}'.format(len(dataset_train)))

for epoch_num in range(parser.epochs):

    retinanet.train()
    retinanet.module.freeze_bn()

    epoch_loss = []

    for iter_num, data in enumerate(dataloader_train):
        try:
            optimizer.zero_grad()

            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
            continue

    if parser.dataset == 'coco':

        print('Evaluating dataset')

        coco_eval.evaluate_coco(dataset_val, retinanet)

    elif parser.dataset == 'csv' and parser.csv_val is not None:

        print('Evaluating dataset')

        mAP = csv_eval.evaluate(dataset_val, retinanet)

    scheduler.step(np.mean(epoch_loss))

torch.save(retinanet.module.state_dict(), './{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

!cp ./testcsv_retinanet_1.pt /content/drive/My\ Drive/pytorch_hackathon