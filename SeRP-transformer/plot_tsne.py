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

from sklearn.manifold import TSNE
import seaborn as sns

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

# model_name = 'serp'
model_name = 'vasp'

if model_name == 'serp':
    model = Point_SERP().to(device)
    path = 'models/pre-trained/tr_serp/model.pth'
    img_folder = f'images/reconstructed/tr_serp/tsne'

elif model_name == 'vasp':
    model = VASP().to(device)
    path = 'models/pre-trained/tr_vasp/model.pth'
    img_folder = f'images/reconstructed/tr_vasp/tsne_zq'

os.makedirs(img_folder, exist_ok=True)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['best_model_wts'])
del checkpoint

num_cats = 0
latents = None
labels = []
for tax_id in label_ids.keys():

    condtn = tax_id[1:]
    model_df = pc_data.query(f'(synsetId == {condtn})').sample(n=50)

    for idx, row in model_df.iterrows():

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

        latent = model(norm_pc_corrupt, norm_pc_sampled, y_corrupt, eval=True)

        latent_np = latent.detach().cpu().numpy()
        if isinstance(latents, type(None)):
            latents = np.copy(latent_np)
        else:
            latents = np.concatenate((latents, np.copy(latent_np)), axis=0)

        print('latents:', latents.shape)

        labels.append(num_cats)

    num_cats += 1
    if num_cats == 5:
        break

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(latents)

plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=labels)
path = os.path.join(img_folder, 'tsne_2d.png')
plt.savefig(path)

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(latents)

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(projection='3d')
# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=tsne_results[:,0], 
    ys=tsne_results[:,1], 
    zs=tsne_results[:,2], 
    c=labels, 
    cmap='tab10'
)

ax.set_xlabel('tsne-one')
ax.set_ylabel('tsne-two')
ax.set_zlabel('tsne-three')

path = os.path.join(img_folder, 'tsne_3d.png')
plt.savefig(path)
