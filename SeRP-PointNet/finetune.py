import numpy as np
import random
import torch
import os, argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_utils import ShapeNet, ModelNet40
from data_utils import PointcloudScaleAndTranslate
import wandb
from tqdm import tqdm
from time import time
from models.pointnet import SerpPointNetClassifier, SerpPointNet


def run_model(model, dataloader, optimizer=None, criterion=None, mode='train', use_wandb=False):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    n_batches = len(dataloader)

    avg_cls_loss = 0.
    avg_acc = 0.

    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(tqdm(dataloader, ascii=True)):

        if mode == 'train':
            optimizer.zero_grad()
            logits = model(pc_sampled.cuda())
            loss = criterion(logits, tax_id.cuda().long())
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({
                    "Train cls loss in-batch": loss.item()
                })

        else:
            with torch.no_grad():
                logits = model(pc_sampled.cuda())
                loss = criterion(logits, tax_id.cuda().long())

        avg_cls_loss += loss.item()

        pred = logits.argmax(-1).detach().cpu().numpy()
        labels = tax_id.detach().cpu().numpy()
        matches = (pred == labels).sum()
        avg_acc += matches
        # break

    avg_cls_loss /= n_batches
    avg_acc /= (n_batches * args.batch_size)

    return avg_cls_loss, avg_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, required=True) # Available options: vq, pointnet, transformers
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--backbone_ckpt', type=str, default=None)
    parser.add_argument('--dataroot', type=str, default="../data")
    parser.add_argument('--dataset', type=str, default='modelnet')

    args = parser.parse_args()

    timestamp = int(time())
    if args.backbone_ckpt:
        ckpt = args.backbone_ckpt.split("/")[1]
    else:
        ckpt = None
    what_is_running = f'finetune_{args.model}_epochs{args.epochs}_{args.dataset}_{ckpt}_{timestamp}'
    if args.use_wandb:
        wandb.init(name=what_is_running, project="674", entity="682_dbdc_dm")
        wandb.config.update(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')

    train_transforms = transforms.Compose(
        [
            PointcloudScaleAndTranslate(),
        ]
    )

    if args.dataset == 'shapenet':
        dataset = ShapeNet('train', train_transforms, dataroot=args.dataroot)
        trainDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataset = ShapeNet('test', dataroot=args.dataroot)
        testDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = 55

    elif args.dataset == 'modelnet':
        dataset = ModelNet40('train', dataroot=args.dataroot)
        trainDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataset = ModelNet40('test', dataroot=args.dataroot)
        testDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = 40

    backbone_model = SerpPointNet(finetuning= True)
    if args.backbone_ckpt is not None:
        checkpoint = torch.load(args.backbone_ckpt)
        backbone_model.load_state_dict(checkpoint['model_wts'])
        print(f'Model weights loaded from {args.backbone_ckpt}')

    model = SerpPointNetClassifier(backbone =backbone_model, num_label = num_classes).to(device)


    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=args.epochs,
                                                           eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss()

    max_acc = 0.

    for epoch in range(args.epochs):
        avg_cls_loss, avg_acc = run_model(model=model, dataloader=trainDataloader, optimizer=optimizer,criterion= criterion, mode='train', use_wandb = args.use_wandb)
        if args.use_wandb:
            wandb.log({"Train Classification Loss": avg_cls_loss,
                       "Train Accuracy": avg_acc})
        if (epoch + 1) % 1 == 0:
            print("Testing")
            avg_cls_loss, avg_acc = run_model(model= model, dataloader=testDataloader, criterion=criterion,mode='test', use_wandb = args.use_wandb)
            if args.use_wandb:
                wandb.log({"Test Classification Loss": avg_cls_loss,
                           "Test Accuracy": avg_acc})

            print("Test Avg Accuracy: ", avg_acc)
            if max_acc < avg_acc:
                max_acc = avg_acc

                ckpt = {
                    'model_wts': model.state_dict(),
                    'max_acc': max_acc
                }

                path = os.path.join("./saved_models/finetuned/", f'{epoch}_{what_is_running}.pth')
                torch.save(ckpt, path)
        scheduler.step()


