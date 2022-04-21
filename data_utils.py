from torch.utils import data
import pandas as pd
import os, argparse, random
import numpy as np
import torch, cv2 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def process_data():

    label_ids = {}
    num_classes = 0

    dataroot = '/mnt/nfs/work1/huiguan/siddhantgarg/datasets/ShapeNet55/ShapeNet55/shapenet_pc'
    all_csv = '/mnt/nfs/work1/huiguan/siddhantgarg/datasets/ShapeNet55/ShapeNetCore.v2/all.csv'
    df = pd.read_csv(all_csv, sep=',')
    for index, row in df.iterrows():
        path = f'{dataroot}/0{row["synsetId"]}-{row["modelId"]}.npy'
        # if not os.path.exists(path):
        #     print(f'{path}')
        #     df.drop(index, inplace=True)

        lab = f'0{row["synsetId"]}'
        if lab not in label_ids.keys():
            label_ids[lab] = num_classes
            num_classes += 1
    print('num_classes:', num_classes)

    torch.save(label_ids, 'label_ids.pth')

    # train_split = df[df['split']=='train']
    # val_split = df[df['split']=='val']

    # path = '/mnt/nfs/work1/huiguan/siddhantgarg/datasets/ShapeNet55/ShapeNet55/train_split.csv'
    # train_split.to_csv(path, index=False, sep=',')

    # path = '/mnt/nfs/work1/huiguan/siddhantgarg/datasets/ShapeNet55/ShapeNet55/val_split.csv'
    # val_split.to_csv(path, index=False, sep=',')
# process_data()

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            # pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(xyz2).float()
            
        return pc



class ShapeNet(Dataset):
    def __init__(self, mode='train', transform=None, normalize=True, return_full=False):

        self.dataroot = '/mnt/nfs/work1/huiguan/siddhantgarg/datasets/ShapeNet55/ShapeNet55/shapenet_pc'
        
        if mode =='train':
            path = '/mnt/nfs/work1/huiguan/siddhantgarg/datasets/ShapeNet55/ShapeNet55/train_split.csv'
            self.pc_data = pd.read_csv(path, sep=',')
        else:
            path = '/mnt/nfs/work1/huiguan/siddhantgarg/datasets/ShapeNet55/ShapeNet55/val_split.csv'
            self.pc_data = pd.read_csv(path, sep=',')
        
        self.label_ids = torch.load('/mnt/nfs/work1/huiguan/siddhantgarg/multitask_pruning/project/serp/label_ids.pth')
        self.return_full = return_full
        self.normalize = normalize
        self.npoints = 1024 
        self.ncenters = 20
        self.n_neighbours = 20
        
        self.transform = transform

    def normalize_pc(self, points):

        center = np.mean(points, axis=0)
        points -= center
        max_norm = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / max_norm 
        return points 

    def random_sample(self, points):
        sample = np.arange(len(points))
        np.random.shuffle(sample)
        points_sampled = points[sample[:self.npoints]]
        return points_sampled 

    def get_centers(self, points):
        sample = np.arange(len(points))
        np.random.shuffle(sample)
        centers_idx = sample[:self.ncenters]
        return centers_idx
    
    def get_neighbors(self, points, centers_idx):
        centers = points[centers_idx]
        centers = np.expand_dims(centers, axis=1)

        dist = centers - points 
        dist = dist**2
        dist = np.sum(dist, axis=-1)

        neigh_idx = np.argsort(dist, axis=-1)[:,:self.n_neighbours]
        neigh_idx = np.reshape(neigh_idx, -1)
        neigh_idx = np.unique(neigh_idx) 

        return neigh_idx

    def corrupt_pc(self, points):
        centers_idx = self.get_centers(points)
        neigh_idx = self.get_neighbors(points, centers_idx)

        y = np.zeros(len(points))
        y[centers_idx] = 1
        y[neigh_idx] = 1 # corrupted data points

        jitter = np.random.normal(0, 0.05, size=(len(points), 3)) 
        # print('jitter:', jitter.shape)     
        # print('y:', y.reshape((-1,1)).shape)     
        jitter *= y.reshape((-1,1)) # corrupt only labeled 1 points 

        # print('max_jitter:',np.amax(jitter))
        # print('max_jitter:',np.amax(jitter))
        # print('#corrupted:', np.sum(y))

        corrupted = points + jitter
        return corrupted, y 

    def __len__(self):
        return self.pc_data.shape[0]

    def __getitem__(self, idx):
        
        row = self.pc_data.iloc[idx]
        path = f'{self.dataroot}/0{row["synsetId"]}-{row["modelId"]}.npy'

        tax_id = f'0{row["synsetId"]}'
        label = self.label_ids[tax_id]

        pc_full = np.load(path).astype(np.float32)

        if self.transform:
            pc_full = torch.from_numpy(pc_full).float()
            pc_full = torch.unsqueeze(pc_full, 0)
            pc_full = self.transform(pc_full)
            pc_full = torch.squeeze(pc_full)
            pc_full = pc_full.detach().cpu().numpy()

        pc_sampled = self.random_sample(pc_full)
        pc_corrupt, y_corrupt = self.corrupt_pc(pc_sampled)
        
        if self.normalize:    
            pc_corrupt = self.normalize_pc(pc_corrupt)
            pc_sampled = self.normalize_pc(pc_sampled) 

        pc_corrupt = torch.from_numpy(pc_corrupt).float()
        pc_sampled = torch.from_numpy(pc_sampled).float()

        if self.return_full:
            pc_full = torch.from_numpy(pc_full).float()
            return label, pc_full, pc_sampled, pc_corrupt, y_corrupt
        else:
            return label, pc_sampled, pc_corrupt, y_corrupt
             
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

def vis_pc():

    roll_pitch = {"02691156" : (90, 135), '04379243' : (30, 30), '03642806' : (30, -45), '03467517' : (0, 90), 
                    '03261776' : (0, 75), '03001627' : (30, -45)}

    label_ids = torch.load('/mnt/nfs/work1/huiguan/siddhantgarg/multitask_pruning/project/serp/label_ids.pth')
    roll_pitch = {label_ids[key] : v for key, v in roll_pitch.items()}

    dataset = ShapeNet('train', False, True)
    trainDataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, (tax_id, pc_full, pc_sampled, pc_corrupt, y_corrupt) in enumerate(trainDataloader):
        
        tax_id = tax_id[0]
        pc = pc_full[0].detach().cpu().numpy()

        if tax_id in roll_pitch.keys():
            r, l = roll_pitch[tax_id]
        else:
            r,l = 0, 0    
        
        img = get_ptcloud_img(pc, r, l)
        path = f'images/{tax_id}_full.png'
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))    

        pc = pc_sampled[0].detach().cpu().numpy()

        if tax_id in roll_pitch.keys():
            r, l = roll_pitch[tax_id]
        else:
            r,l = 0, 0    
        
        img = get_ptcloud_img(pc, r, l)
        path = f'images/{tax_id}_sampled.png'
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))    

        pc = pc_corrupt[0].detach().cpu().numpy()

        if tax_id in roll_pitch.keys():
            r, l = roll_pitch[tax_id]
        else:
            r,l = 0, 0    
        
        img = get_ptcloud_img(pc, r, l)
        path = f'images/{tax_id}_corrupt.png'
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))    
        
        if idx==10:
            break