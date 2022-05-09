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
from Point_MAE.extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function, Variable

from data_utils import ShapeNet
from models import *

class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2

        # print('x_expanded:', x_expanded.shape)
        
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb
        # print('num_arb_dims:', num_arbitrary_dims, emb_expanded.shape)

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        
        diff = x_expanded - emb_expanded
        # print('diff:', diff.shape)
        # print('dist:', dist.shape)

        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        # print('shifted:', shifted_shape)

        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])
        # print('result:', result.shape)
        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


'''
VASP: Vector-quantized Autoencoder for Self-supervised representation learning
of Point clouds. 
'''
class VASP(nn.Module):

    def __init__(self, encoder_only=False):
        super().__init__()
        self.encoder_only = encoder_only
        
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6 

        # patch embeddings
        self.embedding_dims =  384
        self.patch_embedding = PointNet(encoder_channel = self.embedding_dims)

        self.encoder_pos_embed = PositionEmbedding(self.trans_dim)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.serp_encoder = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        
        self.num_latents = 512
        self.emb = NearestEmbed(self.num_latents, self.embedding_dims)
        
        self.group_size = 32
        self.num_group = 64
        self.drop_path_rate = 0.1
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        
        self.decoder_pos_embed = PositionEmbedding(self.trans_dim)

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
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )
        self.discriminator_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 2*self.group_size, 1)
        )

        self.vq_coef = 1
        self.comit_coef = 0.25

        self.loss = 'cdl2'
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

    def forward(self, pts, gt_points, fake_labels):
        # print('gt_points:', gt_points.shape)
        # print('pts:', pts.shape)

        neighborhood, center, gt_neighborhood, grp_labels = self.group_divider(pts, gt_points, fake_labels)

        # print('fps done')

        patch_tokens = self.patch_embedding(neighborhood) # B G C 

        encoder_pos = self.encoder_pos_embed(center)
        
        # print('pos embed')

        x_encoder = self.serp_encoder(patch_tokens, encoder_pos) # latent_z representation

        if self.encoder_only:
            return x_encoder

        # print('x_encoder:', x_encoder.shape)

        # print('discretizing')
        # z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
        z_q, _ = self.emb(x_encoder.transpose(1,2), weight_sg=True)
        # emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)
        emb, _ = self.emb(x_encoder.transpose(1,2).detach())

        # print('z_q:', z_q.shape)
        # print('emb:', emb.shape)

        z_q = z_q.transpose(1,2)
        emb = emb.transpose(1,2)

        decoder_pos = self.decoder_pos_embed(center)
        x_rec = self.serp_decoder(z_q, decoder_pos) # [B, G, 384]

        # print('x-decoded')
        # print('x_rec:', x_rec.shape)

        B, M, C = x_rec.shape
        tmp = x_rec.transpose(1, 2)  # [bs, 384, num_groups]
        rebuild_points = self.reconstruction_head(tmp) # [bs, 3*grp_size, num_groups]
        rebuild_points = rebuild_points.transpose(1, 2)  #   [bs, num_groups, 3*grp_size]
        rebuild_points = rebuild_points.reshape(B, -1, 3)

        # print('rebuilt')
        
        iscorrupt = self.discriminator_head(tmp) # [bs, 2*grp_size, num_groups]
        iscorrupt = iscorrupt.transpose(1, 2)  #   [bs, num_groups, 2*grp_size]
        iscorrupt = iscorrupt.reshape(B, -1, 2) # [bs * num_groups * grp_size, 2]
        
        # print('classified')

        grp_labels = grp_labels.view(B, -1, 1)

        rec_loss = self.loss_func(gt_neighborhood.view(B, -1, 3), rebuild_points)
        classifier_loss = self.discriminator_loss(grp_labels, iscorrupt)

        vq_loss = F.mse_loss(emb, x_encoder.detach())
        commit_loss = F.mse_loss(x_encoder, emb.detach())

        return rec_loss, classifier_loss, vq_loss, commit_loss

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = VASP().to(device)
# dataset = ShapeNet('train')
# trainDataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(trainDataloader):
#     tax_id = tax_id[0]
#     print('pc_cor:', pc_corrupt.shape)
#     rec_loss, classifier_loss, vq_loss, commit_loss = model(pc_corrupt.cuda(), pc_sampled.cuda(), y_corrupt.cuda())

#     print('rec_loss:', rec_loss)
#     print('classifier_loss:', classifier_loss)
#     print('vq_loss:', vq_loss)
#     print('commit_loss:', commit_loss)
#     break