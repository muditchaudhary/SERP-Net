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
from Point_MAE.extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.scheduler import CosineLRScheduler


from data_utils import ShapeNet
from data_utils import PointcloudScaleAndTranslate
from models import Point_SERP
from vq_vae import VASP 

parser = argparse.ArgumentParser()

parser.add_argument('--logs_dir', type=str, default='models/pretrain')

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--initial_epochs', type=int, default=10)
parser.add_argument('--step_per_update', type=int, default=1)

parser.add_argument('--classifier_coeff', type=float, default=0.25)
parser.add_argument('--commit_coef', type=float, default=0.25)
parser.add_argument('--emb_coef', type=float, default=0.25)

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
trainDataloader = DataLoader(dataset, batch_size=128, shuffle=True)

dataset = ShapeNet('test')
testDataloader = DataLoader(dataset, batch_size=128, shuffle=True)

if args.use_vq:
    model = VASP().to(device)
    val_model = VASP().to(device)
else:
    model = Point_SERP().to(device)
    val_model = Point_SERP(encoder_only=True).to(device)

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
            losses = model(pc_corrupt.cuda(), pc_sampled.cuda(), y_corrupt.cuda())
            rec_loss, classifier_loss, vq_loss, commit_loss = losses

            loss = rec_loss 
            loss += args.classifier_coeff * classifier_loss
            loss += args.emb_coef * vq_loss 
            loss += args.comit_coef * commit_loss

            avg_rec_loss += rec_loss.item()
            avg_cls_loss += classifier_loss.item()
            avg_vq_loss += vq_loss.item()
            avg_commit_loss += commit_loss.item()

            logprint(f'{idx+1}/{n_batches} ')
            logprint(f'rec_loss: {avg_rec_loss/(idx+1) :.6f} ')
            logprint(f'cls_loss: {avg_cls_loss/(idx+1) :.6f} ')
            logprint(f'vq_loss: {avg_vq_loss/(idx+1) :.6f} ')
            logprint(f'commit_loss: {avg_commit_loss/(idx+1) :.6f}\n')

        else:
            rec_loss, classifier_loss = model(pc_corrupt.cuda(), pc_sampled.cuda(), y_corrupt.cuda())
            loss = rec_loss + args.classifier_coeff * classifier_loss
 
            avg_rec_loss += rec_loss.item()
            avg_cls_loss += classifier_loss.item()
            
            logprint(f'{idx+1}/{n_batches} ')
            logprint(f'rec_loss: {avg_rec_loss/(idx+1) :.6f} ')
            logprint(f'cls_loss: {avg_cls_loss/(idx+1) :.6f} ')
            
        loss.backward()
        optimizer.step()        

        break

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
    sd = model.state_dict()
    val_model.load_state_dict(sd)

    val_model.eval()

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(trainDataloader):

        feats_b = val_model(pc_sampled.cuda(), pc_sampled.cuda(), y_corrupt.cuda())
        label_b = tax_id

        train_features.append(feats_b)
        train_labels.append(label_b)

        if idx == 3:
            break
    
    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(testDataloader):

        feats_b = val_model(pc_sampled)
        label_b = tax_id

        test_features.append(feats_b)
        test_labels.append(label_b)

        if idx == 3:
            break

        
    train_features = torch.cat(train_features, dim=0)
    train_label = torch.cat(train_label, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_label = torch.cat(test_label, dim=0)

    svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_labels.data.cpu().numpy(), test_features.data.cpu().numpy(), test_labels.data.cpu().numpy())

    return svm_acc


max_svm_acc = 0.

for epoch in range(args.epochs):
    
    avg_rec_loss, avg_cls_loss = train_model(trainDataloader)
    scheduler.step()

    logprint(f'epoch:{epoch+1}/{args.epochs} rec_loss: {avg_rec_loss :.4f} cls:{avg_cls_loss :.4f}\n')

    svm_acc = validate()

    logprint(f'epoch:{epoch+1}/{args.epochs} svm_acc:{svm_acc}\n')

    if max_svm_acc < svm_acc:

        max_svm_acc = svm_acc

        ckpt = {
            'best_model_wts' : model.state_dict(),
            'max_svm_acc' : max_svm_acc
        }

        path = os.path.join(args.logs_dir, 'model.pth')
        torch.save(ckpt, path)
