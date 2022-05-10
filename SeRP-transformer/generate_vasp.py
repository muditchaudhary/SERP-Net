import sys 
sys.path.append('../')
from torch.utils import data
import pandas as pd
import os, argparse, random, sys
import numpy as np
import torch, cv2 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from serp_transformer import Point_SERP
from vq_vae import VASP 

from utils import corrupt_pc, get_ptcloud_img, random_sample, normalize_pc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

roll_pitch = {"02691156" : (90, 135), '04379243' : (30, 30), '03642806' : (30, -45), '03467517' : (0, 90), 
                    '03261776' : (0, 75), '03001627' : (30, -45)}

label_ids = torch.load('../data/ShapeNet55/ShapeNet55/label_ids.pth')

roll_pitch = {label_ids[key] : v for key, v in roll_pitch.items()}

dataroot = '../data/ShapeNet55/ShapeNet55/shapenet_pc'

path = '../data/ShapeNet55/ShapeNet55/train_split.csv'
pc_data = pd.read_csv(path, sep=',')

npoints = 1024 
ncenters = 20
n_neighbours = 20

model = VASP().to(device)

path = 'models/pre-trained/tr_vasp/model.pth'

checkpoint = torch.load(path)
model.load_state_dict(checkpoint['best_model_wts'])

del checkpoint

# print(pc_data.head())

def write_reconstructed(epoch):

    for tax_id in label_ids.keys():

        condtn = tax_id[1:]
        model_df = pc_data.query(f'(synsetId == {condtn})').sample(n=5)
        print(f'reconstructing {tax_id}\n')

        for idx, row in model_df.iterrows():

            img_folder = f'images/reconstructed/tr_vasp/train/{epoch}'
            os.makedirs(img_folder, exist_ok=True)

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

            # sample = np.arange(len(rebuild_points))
            # np.random.shuffle(sample)
            # rebuild_points = rebuild_points[sample[:npoints]]

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
        

write_reconstructed(300)
