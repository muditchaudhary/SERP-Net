import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, argparse
from collections import abc
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.scheduler import CosineLRScheduler
from sklearn.svm import LinearSVC

from data_utils import ShapeNet, ModelNet40
from data_utils import PointcloudScaleAndTranslate

from transformer_finetune import TransformerFinetune

parser = argparse.ArgumentParser()

parser.add_argument('--logs_dir', type=str, default='evaluation')

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--initial_epochs', type=int, default=10)
parser.add_argument('--step_per_update', type=int, default=1)

parser.add_argument('--classifier_coeff', type=float, default=0.25)
parser.add_argument('--commit_coef', type=float, default=0.25)
parser.add_argument('--emb_coef', type=float, default=0.25)

parser.add_argument('--prev_ckpt', type=str, default=None)
parser.add_argument('--finetuned_model', type=str, required=None)

parser.add_argument('--num_classes', type=int, default=40)

parser.add_argument('--dataset', type=str, default='modelnet')

args = parser.parse_args()

if not os.path.exists(args.logs_dir):
    os.mkdir(args.logs_dir)

logs_file = os.path.join(args.logs_dir, 'logs.txt')

def logprint(log):
    print(log, end='')
    with open(logs_file, 'a') as f:
        f.write(log) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')

train_transforms = transforms.Compose(
    [
        PointcloudScaleAndTranslate(),
    ]
)

if args.dataset == 'shapenet':
    dataset = ShapeNet('train', train_transforms)
    trainDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataset = ShapeNet('test')
    testDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    num_classes = 55

elif args.dataset == 'modelnet':
    dataset = ModelNet40('train')
    trainDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataset = ModelNet40('test')
    testDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    num_classes = 40

model = TransformerFinetune(num_classes).to(device)

max_acc = 0.

if not isinstance(args.finetuned_model, type(None)):
    checkpoint = torch.load(args.finetuned_model)

    model.load_state_dict(checkpoint['best_model_wts'])
    max_acc = checkpoint['max_acc']

    logprint('fine-tuned model uploaded!\n')
    logprint(f'best_acc: {checkpoint["max_acc"]}\n')
    del checkpoint

if not isinstance(args.prev_ckpt, type(None)):
    checkpoint = torch.load(args.prev_ckpt)
    pretrained_state_dict = checkpoint['best_model_wts']
    for k in model.state_dict().keys():
        if k in pretrained_state_dict.keys():
            v = pretrained_state_dict[k]
            model.state_dict()[k].data.copy_(v)

    del checkpoint
    logprint(f'wts loaded from {args.prev_ckpt}!\n')

optimizer = torch.optim.AdamW(model.parameters(),
                lr = args.learning_rate, 
                weight_decay=args.weight_decay,
            )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=args.epochs,
                                                       eta_min=1e-6)

criterion = nn.CrossEntropyLoss()

def run_model(dataloader, mode='train'):
    if mode=='train':
        model.train()
    else:
        model.eval()

    n_batches = len(dataloader) 

    avg_cls_loss = 0.
    avg_acc = 0. 

    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(dataloader):

        if mode == 'train':
            optimizer.zero_grad()   
            logits = model(pc_sampled.cuda())
            loss = criterion(logits, tax_id.cuda().long())
            loss.backward()
            optimizer.step()     

        else:
            with torch.no_grad():
                logits = model(pc_sampled.cuda())
                loss = criterion(logits, tax_id.cuda().long())

        avg_cls_loss += loss.item()

        pred = logits.argmax(-1).detach().cpu().numpy()
        labels = tax_id.detach().cpu().numpy()
        matches = (pred == labels).sum() 
        avg_acc += matches
        
        logprint(f'{idx+1}/{n_batches} ')
        logprint(f'cls_loss: {avg_cls_loss/(idx+1) :.6f} ')
        logprint(f'accuracy: {avg_acc/(args.batch_size *(idx+1)) :.6f}\n')

        # break

    avg_cls_loss /= n_batches
    avg_acc /= (n_batches * args.batch_size)

    return avg_cls_loss, avg_acc


logprint('============\n')
logprint('testing')
logprint('============\n')

avg_cls_loss, avg_acc = run_model(testDataloader, 'test')
logprint(f'cls_loss: {avg_cls_loss :.4f} acc:{avg_acc :.4f}\n')
