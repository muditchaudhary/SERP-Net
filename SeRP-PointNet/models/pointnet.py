from __future__ import print_function
import sys

sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from data_utils import ShapeNet
from torch.utils.data import Dataset, DataLoader


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            global_feature = x
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, global_feature


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class SerpPointNet(nn.Module):
    def __init__(self, output_feat_dim=128, feature_transform=False, rec_loss="mse", learn_delta=False,
                 weight_lambda=0.75, learn_diff = False, finetuning = False):
        super(SerpPointNet, self).__init__()

        self.finetuning = finetuning
        self.learn_delta = learn_delta
        self.learn_diff = learn_diff
        self.output_feat_dim = output_feat_dim
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.output_feat_dim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.cls_head = torch.nn.Conv1d(self.output_feat_dim, 2, 1)
        self.reconstruction_head = torch.nn.Conv1d(self.output_feat_dim, 3, 1)
        self.weight_lambda = weight_lambda

        self.cls_loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([0.25, 0.75]))

        if rec_loss == "cdl1":
            self.rec_loss_func = ChamferDistanceL1().cuda()
        elif rec_loss == 'cdl2':
            self.rec_loss_func = ChamferDistanceL2().cuda()
        elif rec_loss == 'mse':
            self.rec_loss_func = torch.nn.MSELoss().cuda()
        elif rec_loss == "l1":
            self.rec_loss_func = torch.nn.L1Loss().cuda()
        elif rec_loss == "smoothL1":
            self.rec_loss_func = torch.nn.SmoothL1Loss().cuda()
        elif rec_loss == "weightedL2":
            self.rec_loss_func = self.weightedL2Loss
        else:
            raise NotImplementedError

        self.rec_loss = rec_loss

    def cls_loss(self, logits, labels):
        loss = self.cls_loss_func(logits, labels)
        return loss

    def weightedL2Loss(self, pred, target, cls_labels):
        weights = torch.where(cls_labels == 1, 1+self.weight_lambda, 1.0)
        return torch.mean(weights * torch.sum((pred - target) ** 2,dim=2).reshape(-1))

    def weightedL1Loss(self, pred, target, cls_labels):
        weights = torch.where(cls_labels == 1, 1+self.weight_lambda, 1.0)
        return torch.mean(weights * torch.sum((pred - target),dim=2).reshape(-1))

    def reconstruction_loss(self, reconstructed_pts, gt_points):
        if self.rec_loss in ["cdl1", "cdl2"]:
            rec_loss = self.rec_loss_func(gt_points, reconstructed_pts)
        else:
            rec_loss = self.rec_loss_func(reconstructed_pts, gt_points)
        return rec_loss

    def forward(self, x, rec_labels=None, cls_labels=None, reconstruct=False):
        x = x.transpose(2, 1)
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        feats, trans, trans_feat, global_feature = self.feat(x)

        if self.finetuning:
            return global_feature

        feats = F.relu(self.bn1(self.conv1(feats)))
        feats = F.relu(self.bn2(self.conv2(feats)))
        feats = F.relu(self.bn3(self.conv3(feats)))
        per_point_feature = self.conv4(feats)  # (N, output_feat_dim, num_points)

        if self.training:
            cls_logits = self.cls_head(per_point_feature)
            cls_logits = cls_logits.transpose(2, 1)
            cls_logits = cls_logits.reshape((-1, 2))
            cls_labels = cls_labels.reshape(-1).long()
            cls_loss = self.cls_loss(cls_logits, cls_labels)
            rec_points = self.reconstruction_head(per_point_feature)
            rec_points = rec_points.transpose(2, 1)

            if self.learn_delta:

                #gt_delta = rec_labels - x.transpose(2, 1)
                #pred_delta = rec_points
                rec_points = rec_points + x.transpose(2,1)
                #(Batchsize, point_cloud_idx, x,y,z)

                if self.rec_loss in ["mse", "l1", "smoothL1", "cdl1", "cdl2"]:
                    rec_loss = self.reconstruction_loss(rec_points, rec_labels)
                elif self.rec_loss == "weightedL2":
                    rec_loss = self.weightedL2Loss(pred_delta, gt_delta, cls_labels)
                else:
                    raise NotImplementedError
            elif self.learn_diff:
                gt_delta = rec_labels - x.transpose(2, 1)
                pred_delta = rec_points
                #rec_points = rec_points + x.transpose(2, 1)
                # (Batchsize, point_cloud_idx, x,y,z)

                if self.rec_loss in ["mse", "l1", "smoothL1", "cdl1", "cdl2"]:
                    rec_loss = self.reconstruction_loss(pred_delta, gt_delta)
                elif self.rec_loss == "weightedL2":
                    rec_loss = self.weightedL2Loss(pred_delta, gt_delta, cls_labels)
                else:
                    raise NotImplementedError

            else:

                rec_loss = self.reconstruction_loss(rec_points, rec_labels)

            return rec_loss, cls_loss

        elif reconstruct:
            rec_points = self.reconstruction_head(per_point_feature)
            if self.learn_delta or self.learn_diff:
                pred_delta = rec_points.transpose(2, 1)
                rec_points = pred_delta + x.transpose(2, 1)
            else:
                rec_points = rec_points.transpose(2, 1)

            return rec_labels, rec_points
        else:
            return global_feature, per_point_feature

class SerpPointNetClassifier(nn.Module):
    def __init__(self, backbone, num_label, global_feature_dim = 1024):
        super(SerpPointNetClassifier, self).__init__()
        self.backbone = backbone
        self.num_label = num_label
        self.cls_head = torch.nn.Linear(global_feature_dim, self.num_label)

    def forward(self, x):
        out = self.backbone(x)
        out = self.cls_head(out)

        return out

class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat, global_feature = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SerpPointNet(rec_loss="mse")
    model = model.to(device)
    # data = torch.randn((2,5,3))
    # #dataT = data.transpose(2,1)
    # labels = target = torch.empty((2,5), dtype=torch.long).random_(2)

    # y = model(dataT, labels,0)

    dataset = ShapeNet('train')
    trainDataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(trainDataloader):
        rec_loss, classifier_loss = model(pc_corrupt.cuda(), pc_sampled.cuda(), y_corrupt.cuda())
        print(rec_loss)
        print(classifier_loss)
        break
    # sim_data = Variable(torch.rand(32,3,2500))
    # trans = STN3d()
    # out = trans(sim_data)
    # print('stn', out.size())
    # print('loss', feature_transform_regularizer(out))
    #
    # sim_data_64d = Variable(torch.rand(32, 64, 2500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('stn64d', out.size())
    # print('loss', feature_transform_regularizer(out))
    #
    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('global feat', out.size())
    #
    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())
    #
    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())
    #
    # seg = PointNetDenseCls(k = 3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())
