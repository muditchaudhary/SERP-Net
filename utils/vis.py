import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from data_utils import ShapeNet
from torch.utils.data import Dataset, DataLoader
import numpy as np


def normalize_pc(points):
    center = np.mean(points, axis=0)
    points -= center
    max_norm = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / max_norm
    return points

def denormalize_pc(unnormalized_pts, rec_points):
    center = np.mean(unnormalized_pts, axis=0)
    max_norm = np.max(np.sqrt(np.sum(unnormalized_pts ** 2, axis=1)))
    rec_points = rec_points * max_norm
    rec_points += center
    return rec_points


def get_ptcloud_img(ptcloud, roll, pitch):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    # ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax = fig.add_subplot(projection='3d')

    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(roll, pitch)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=y, cmap='jet')

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    return img

def vis_pc(model):
    roll_pitch = {"02691156": (90, 135), '04379243': (30, 30), '03642806': (30, -45), '03467517': (0, 90),
                  '03261776': (0, 75), '03001627': (30, -45)}

    label_ids = torch.load('./data/ShapeNet55/ShapeNet55/label_ids.pth')
    roll_pitch = {label_ids[key]: v for key, v in roll_pitch.items()}

    dataset = ShapeNet('train', normalize=True, return_full =True)
    print(dataset.normalize)
    trainDataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, (tax_id, pc_full, pc_sampled, pc_corrupt, y_corrupt, pc_sampled_unnormalized, pc_corrupt_unnormalized) in enumerate(trainDataloader):

        tax_id = tax_id[0]
        pc = pc_full[0].detach().cpu().numpy()

        if tax_id in roll_pitch.keys():
            r, l = roll_pitch[tax_id]
        else:
            r, l = 0, 0

        img = get_ptcloud_img(pc, r, l)
        path = f'images/{tax_id}_full.png'
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        pc = pc_sampled_unnormalized[0].detach().cpu().numpy()

        print(pc.shape)

        if tax_id in roll_pitch.keys():
            r, l = roll_pitch[tax_id]
        else:
            r, l = 0, 0

        img = get_ptcloud_img(pc, r, l)
        path = f'images/{tax_id}_sampled.png'
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        pc = pc_corrupt_unnormalized[0].detach().cpu().numpy()

        if tax_id in roll_pitch.keys():
            r, l = roll_pitch[tax_id]
        else:
            r, l = 0, 0

        img = get_ptcloud_img(pc, r, l)
        path = f'images/{tax_id}_corrupt.png'
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        with torch.no_grad():
            reconstructed_pts = model(pc_corrupt.cuda())

        pc = reconstructed_pts.squeeze(dim=0).detach().cpu().numpy()
        pc = denormalize_pc(pc_corrupt_unnormalized.cpu().numpy(), pc)
        print(pc.shape)
        img = get_ptcloud_img(pc, r, l)
        path = f'images/{tax_id}_reconstructed.png'
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if idx == 10:
            break
