import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from torch.utils.data import Dataset, DataLoader

from data_utils import ShapeNet



def fps(data, gt_data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    
    fps_gt_data = pointnet2_utils.gather_operation(gt_data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    
    # print('===FPSStart===')
    # print('fps_idx:', fps_idx.shape)
    # print('fps_data:', fps_data.shape)
    # print('===FPSEnd===')
    
    return fps_data, fps_gt_data


class PointNet(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)



class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, gt_points, labels):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape

        # fps the centers out
        center, gt_center = fps(xyz, gt_points, self.num_group) # B G 3

        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # neighborhood = neighborhood - center.unsqueeze(2)

        gt_neighborhood = gt_points.view(batch_size * num_points, -1)[idx, :]
        gt_neighborhood = gt_neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # gt_neighborhood = gt_neighborhood - gt_center.unsqueeze(2)
        
        grp_labels = labels.view(-1, 1)[idx, :]
        grp_labels = grp_labels.view(batch_size, self.num_group, self.group_size, 1).contiguous()

        return neighborhood, center, gt_neighborhood, gt_center, grp_labels

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x))  # only return the mask tokens predict pixel
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, trans_dim):

        super().__init__()
        self.block = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, trans_dim)
            )
    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, trans_dim):
        super().__init__()
        self.block = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, trans_dim)
            )
    def forward(self, x):
        return self.block(x)

class Point_SERP(nn.Module):

    def __init__(self, loss_type='cdl2'):
        super().__init__()
        
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6 

        # patch embeddings
        self.embedding_dims =  384
        self.patch_embedding = PointNet(encoder_channel = self.embedding_dims)

        self.encoder_pos_embed = PositionEmbedding(self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.serp_encoder = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        
        self.group_size = 32
        self.num_group = 64
        self.drop_path_rate = 0.1

        # self.decoder_pos_embed = PositionEmbedding(self.trans_dim)

        self.decoder_depth = 4
        self.decoder_num_heads = 6
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        
        self.serp_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        # print_log(f'[Point_SERP] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(in_channels=self.trans_dim, out_channels=3*self.group_size, kernel_size=2)
        )
        self.discriminator_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 2*self.group_size, 2)
        )

        self.loss = loss_type
        # loss
        self.build_loss_func(self.loss)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()
    
    def discriminator_loss(self, labels, logits):
        labels = labels.view(-1).long()
        logits = logits.view(-1, 2)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return loss

    def forward(self, pts, gt_points, fake_labels, eval=False, reconstruct=False):
        neighborhood, center, gt_neighborhood, gt_center, grp_labels = self.group_divider(pts, gt_points, fake_labels)
        

        patch_tokens = self.patch_embedding(neighborhood - center.unsqueeze(2)) # B G C 
        pos_embeds = self.encoder_pos_embed(center) # B, G, 384

        cls_tokens = self.cls_token.expand(patch_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(patch_tokens.size(0), -1, -1)

        x = torch.cat((cls_tokens, patch_tokens), dim=1) # B, G+1, 384
        pos = torch.cat((cls_pos, pos_embeds), dim=1) # B, G+1, 384

        x_encoder = self.serp_encoder(x, pos) # [B, G+1, 384]
        if eval:
            concat_f = torch.cat([x_encoder[:, 0], x_encoder[:, 1:].max(1)[0]], dim=-1)
            return concat_f

        # decoder_pos = self.decoder_pos_embed(center) # [B, G, 384] 
        
        x_rec = self.serp_decoder(x_encoder, pos) # [B, G+1, 384]

        B, M, C = x_rec.shape
        x_rec = x_rec.transpose(1, 2)  # [B, 384, G+1]

        rebuild_points = self.reconstruction_head(x_rec) # [B, 3*grp_size, G]
        rebuild_points = rebuild_points.transpose(1, 2)  #   [B, G, 3*grp_size]

        
        if reconstruct:
            rebuild_points = rebuild_points.reshape(B, self.num_group, -1, 3)
            rebuild_points = rebuild_points + gt_center.unsqueeze(2)
            rebuild_points = rebuild_points.reshape(B, -1, 3)
            gt_pc = gt_neighborhood.reshape(B, -1, 3)
            return gt_pc, rebuild_points

        rebuild_points = rebuild_points.reshape(B, -1, 3)

        iscorrupt = self.discriminator_head(x_rec) # [B, 2*grp_size, G]
        iscorrupt = iscorrupt.transpose(1, 2)  #   [B, G, 2*grp_size]
        iscorrupt = iscorrupt.reshape(B, -1, 2) # [B * G * grp_size, 2]
        
        grp_labels = grp_labels.view(B, -1, 1)

        gt_center_normalized = gt_neighborhood - gt_center.unsqueeze(2)

        rec_loss = self.loss_func(gt_center_normalized.view(B, -1, 3), rebuild_points)
        classifier_loss = self.discriminator_loss(grp_labels, iscorrupt)

        return rec_loss, classifier_loss


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = Point_SERP().to(device)
# dataset = ShapeNet('train')
# trainDataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(trainDataloader):
#     print('idx: ', idx, 'input:', pc_sampled.shape)
    # tax_id = tax_id[0]
    # rec_loss, classifier_loss = model(pc_corrupt.cuda(), pc_sampled.cuda(), y_corrupt.cuda())
    # break
