from torch.utils import data
import pandas as pd
import os, argparse, random
import numpy as np
import torch, cv2, sys 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from data_utils import ShapeNet
from data_utils import PointcloudScaleAndTranslate
from serp_transformer import Point_SERP
from vq_vae import VASP 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

roll_pitch = {"02691156" : (90, 135), '04379243' : (30, 30), '03642806' : (30, -45), '03467517' : (0, 90), 
                    '03261776' : (0, 75), '03001627' : (30, -45)}

label_ids = torch.load('../data/ShapeNet55/ShapeNet55/label_ids.pth')

roll_pitch = {label_ids[key] : v for key, v in roll_pitch.items()}

def random_sample(points, npoints):
    sample = np.arange(len(points))
    np.random.shuffle(sample)
    points_sampled = points[sample[:npoints]]
    return points_sampled 

def normalize_pc(points, center=None, max_norm=None):
    if isinstance(center, type(None)):
        center = np.mean(points, axis=0)
    norm_points = points - center.reshape(1, -1)
    if isinstance(max_norm, type(None)):
        max_norm = np.max(np.sqrt(np.sum(points**2, axis=1)))
    norm_points = norm_points / max_norm 
    return norm_points, center, max_norm

def get_centers(points, ncenters):
    sample = np.arange(len(points))
    np.random.shuffle(sample)
    centers_idx = sample[:ncenters]
    return centers_idx

def get_neighbors(points, centers_idx, n_neighbours):
    centers = points[centers_idx]
    centers = np.expand_dims(centers, axis=1)

    dist = centers - points 
    dist = dist**2
    dist = np.sum(dist, axis=-1)

    neigh_idx = np.argsort(dist, axis=-1)[:,:n_neighbours]
    # neigh_idx = np.reshape(neigh_idx, -1)
    # neigh_idx = np.unique(neigh_idx) 

    return neigh_idx

def corrupt_pc(points, ncenters, n_neighbours):
    centers_idx = get_centers(points, ncenters)
    neigh_idx = get_neighbors(points, centers_idx, n_neighbours)

    y = np.zeros(len(points))
    y[centers_idx] = 1
    y[neigh_idx] = 1 # corrupted data points

    center_jitter = np.random.normal(0, 0.03, size=(centers_idx.shape[0], 3)) 
    jitter = np.zeros((len(points), 3))

    for i, epsilon in enumerate(center_jitter):
        nidx = neigh_idx[i]
        jitter[nidx] = epsilon

    corrupted = points + jitter

    return corrupted, y 

def get_ptcloud_img(ptcloud, roll, pitch):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    # ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax = fig.add_subplot(projection='3d')

    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(roll,pitch)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=y, cmap='jet')

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

    plt.close()
    return img

def plot_points(tax_id, path, full, sampled, corrupt, reconstructed):

    if tax_id in roll_pitch.keys():
        r, l = roll_pitch[tax_id]
    else:
        r,l = 0, 0    

    i1= get_ptcloud_img(full, r, l)
    i2 = get_ptcloud_img(sampled, r, l)
    i3 = get_ptcloud_img(corrupt, r, l)
    i4 = get_ptcloud_img(reconstructed, r, l)

    f, axarr = plt.subplots(1,4, figsize=(15, 8))
    axarr[0].imshow(i1)
    axarr[0].axis('off')
    axarr[0].title.set_text('full')

    axarr[1].imshow(i2)
    axarr[1].axis('off')
    axarr[1].title.set_text('sampled')
    
    axarr[2].imshow(i3)
    axarr[2].axis('off')
    axarr[2].title.set_text('corrupted')
    
    axarr[3].imshow(i4)
    axarr[3].axis('off')
    axarr[3].title.set_text('reconstructed')

    plt.axis('off')
    plt.savefig(path)
    plt.close()

def write_reconstructed(epoch):

    for tax_id in label_ids.keys():

        condtn = tax_id[1:]
        model_df = pc_data.query(f'(synsetId == {condtn})').sample(n=5)
        # print(model_df.head())
        print(f'reconstructing {tax_id}\n')

        for idx, row in model_df.iterrows():

            img_folder = f'images/serp/reconstructed/{epoch}'
            if not os.path.exists(img_folder):
                os.mkdir(img_folder)

            path = f'{dataroot}/0{row["synsetId"]}-{row["modelId"]}.npy'
            label = label_ids[tax_id]

            pc_full = np.load(path).astype(np.float32)    
            pc_sampled = random_sample(pc_full, npoints)
            
            pc_corrupt, y_corrupt = corrupt_pc(pc_sampled, ncenters, n_neighbours)

            norm_pc_sampled, mean, max_norm = normalize_pc(pc_sampled)
            norm_pc_corrupt, _, _ = normalize_pc(pc_corrupt, mean, max_norm)
            
            norm_pc_sampled = torch.from_numpy(norm_pc_sampled).float().unsqueeze(0).to(device)
            norm_pc_corrupt = torch.from_numpy(norm_pc_corrupt).float().unsqueeze(0).to(device)
            y_corrupt = torch.from_numpy(y_corrupt).float().unsqueeze(0).to(device)

            gt_points, rebuild_points = model(norm_pc_corrupt, norm_pc_sampled, y_corrupt, reconstruct=True)

            gt_points = gt_points.detach().cpu().numpy()[0]
            rebuild_points = rebuild_points.detach().cpu().numpy()[0]

            mean = mean.reshape(1, -1)

            gt_points *= max_norm
            gt_points += mean 

            rebuild_points *= max_norm
            rebuild_points += mean

            if tax_id in roll_pitch.keys():
                r, l = roll_pitch[tax_id]
            else:
                r,l = 0, 0    

            i1= get_ptcloud_img(pc_full, r, l)
            i2 = get_ptcloud_img(pc_sampled, r, l)
            i3 = get_ptcloud_img(pc_corrupt, r, l)
            i4 = get_ptcloud_img(rebuild_points, r, l)

            f, axarr = plt.subplots(1,4, figsize=(30, 8))
            axarr[0].imshow(i1)
            axarr[0].axis('off')
            axarr[0].title.set_text('full')

            axarr[1].imshow(i2)
            axarr[1].axis('off')
            axarr[1].title.set_text('sampled')
            
            axarr[2].imshow(i3)
            axarr[2].axis('off')
            axarr[2].title.set_text('corrupted')
            
            axarr[3].imshow(i4)
            axarr[3].axis('off')
            axarr[3].title.set_text('reconstructed')

            plt.axis('off')

            path = os.path.join(img_folder, f'{tax_id}_{idx+1}.png')
            plt.savefig(path)
            plt.close()
