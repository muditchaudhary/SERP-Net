import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
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

from data_utils import ShapeNet
from data_utils import PointcloudScaleAndTranslate
# from models import Point_SERP
from serp_transformer import Point_SERP
from vq_vae import VASP 

parser = argparse.ArgumentParser()

parser.add_argument('--logs_dir', type=str, default='models/pretrain')

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--initial_epochs', type=int, default=10)
parser.add_argument('--step_per_update', type=int, default=1)

parser.add_argument('--classifier_coeff', type=float, default=0.0)
parser.add_argument('--commit_coef', type=float, default=0.25)
parser.add_argument('--emb_coef', type=float, default=0.25)
parser.add_argument('--loss_type', type=str, default='cdl2')

parser.add_argument('--prev_ckpt', type=str, default=None)

parser.add_argument('--use_vq', type=bool, default=False)


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

dataset = ShapeNet('train', train_transforms)
trainDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

dataset = ShapeNet('test')
testDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

if args.use_vq:
    model = VASP().to(device)
else:
    model = Point_SERP(loss_type=args.loss_type).to(device)

if not isinstance(args.prev_ckpt, type(None)):
    checkpoint = torch.load(args.prev_ckpt)
    
    for k, v in checkpoint['best_model_wts'].items():   
        if k in model.state_dict().keys():
            model.state_dict()[k].data.copy_(v)

    # model.load_state_dict(checkpoint['best_model_wts'])
    del checkpoint
    logprint(f'wts loaded from {args.prev_ckpt}!\n')

optimizer = torch.optim.AdamW(model.parameters(),
                lr = args.learning_rate, 
                weight_decay=args.weight_decay,
            )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=args.epochs,
                                                       eta_min=1e-6)

def train_model(dataloader):
    model.train()
    n_batches = len(dataloader) 

    avg_rec_loss = 0.
    avg_cls_loss = 0.
    avg_vq_loss = 0.
    avg_commit_loss = 0.

    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(trainDataloader):

        optimizer.zero_grad()
        
        if args.use_vq:
            losses = model(pc_corrupt.to(device), pc_sampled.to(device), y_corrupt.to(device))
            rec_loss, classifier_loss, vq_loss, commit_loss = losses

            loss = rec_loss 
            loss += args.classifier_coeff * classifier_loss
            loss += args.emb_coef * vq_loss 
            loss += args.commit_coef * commit_loss

            avg_rec_loss += rec_loss.item()
            avg_cls_loss += classifier_loss.item()
            avg_vq_loss += vq_loss.item()
            avg_commit_loss += commit_loss.item()

            logprint(f'{idx+1}/{n_batches} ')
            logprint(f'rec_loss: {avg_rec_loss/(idx+1) :.6f} ')
            # logprint(f'cls_loss: {avg_cls_loss/(idx+1) :.6f} ')
            logprint(f'vq_loss: {avg_vq_loss/(idx+1) :.6f} ')
            logprint(f'commit_loss: {avg_commit_loss/(idx+1) :.6f}\n')

        else:
            losses = model(pc_corrupt.to(device), pc_sampled.to(device), y_corrupt.to(device))
            rec_loss, classifier_loss = losses

            loss = rec_loss + args.classifier_coeff * classifier_loss
 
            avg_rec_loss += rec_loss.item()
            avg_cls_loss += classifier_loss.item()
            
            logprint(f'{idx+1}/{n_batches} ')
            logprint(f'rec_loss: {avg_rec_loss/(idx+1) :.6f} ')
            logprint(f'cls_loss: {avg_cls_loss/(idx+1) :.6f}\n')
            
        loss.backward()

        # print(model.discriminator_head[0].weight.grad)
        # print(model.reconstruction_head[0].weight.grad)

        optimizer.step()        

        # if idx==10:
        #     break
        # break

    avg_rec_loss /= n_batches
    avg_cls_loss /= n_batches
    avg_vq_loss /= n_batches
    avg_commit_loss /= n_batches

    if args.use_vq:
        return avg_rec_loss, avg_cls_loss, avg_vq_loss, avg_commit_loss
    else:
        return avg_rec_loss, avg_cls_loss

def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    test_acc = np.sum(test_labels == pred) * 1. / pred.shape[0]
    return test_acc 

def validate():
    model.eval()

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(trainDataloader):

        feats_b = model(pc_sampled.cuda(), pc_sampled.cuda(), y_corrupt.cuda(), eval=True)
        label_b = tax_id

        feats_b = feats_b.cpu().detach()
        # print('feats_b:', feats_b.shape, type(feats_b))

        train_features.append(feats_b)
        train_labels.append(label_b)

        if idx == 10:
            break
    
    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(testDataloader):

        feats_b = model(pc_sampled.cuda(), pc_sampled.cuda(), y_corrupt.cuda(), eval=True)
        label_b = tax_id

        feats_b = feats_b.cpu().detach()

        test_features.append(feats_b)
        test_labels.append(label_b)

        if idx == 10:
            break

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    svm_acc = evaluate_svm(train_features.numpy(), train_labels.numpy(), test_features.numpy(), test_labels.numpy())

    return svm_acc

lowest_rec_loss = 100

for epoch in range(args.epochs):
    
    if args.use_vq:
        avg_rec_loss, avg_cls_loss, avg_vq_loss, avg_commit_loss = train_model(trainDataloader)
        
        logprint(f'epoch:{epoch+1}/{args.epochs} ') 
        logprint(f'rec_loss: {avg_rec_loss :.4f} ')
        # logprint(f'cls:{avg_cls_loss :.4f} ')
        logprint(f'vq_loss: {avg_vq_loss :.4f} ')
        logprint(f'commit:{avg_commit_loss :.4f}\n')
    else:
        avg_rec_loss, avg_cls_loss = train_model(trainDataloader)

        logprint(f'epoch:{epoch+1}/{args.epochs} rec_loss: {avg_rec_loss :.4f} cls:{avg_cls_loss :.4f}\n')

    scheduler.step()

    if avg_rec_loss < lowest_rec_loss:

        lowest_rec_loss = avg_rec_loss

        ckpt = {
            'best_model_wts' : model.state_dict(),
            'lowest_rec_loss' : lowest_rec_loss
        }

        path = os.path.join(args.logs_dir, 'model.pth')
        torch.save(ckpt, path)

    if epoch + 1 % 50 == 0: 
        svm_acc = validate()
        logprint(f'epoch:{epoch+1}/{args.epochs} svm_acc:{svm_acc}\n')

    if epoch + 1 % 25 == 0:

        ckpt = {
            'best_model_wts' : model.state_dict(),
            'lowest_rec_loss' : lowest_rec_loss
        }

        path = os.path.join(args.logs_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(ckpt, path)
