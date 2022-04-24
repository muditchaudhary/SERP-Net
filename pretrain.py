import numpy as np
import random
import torch
import os, argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data_utils import ShapeNet
from data_utils import PointcloudScaleAndTranslate
import wandb
from tqdm import tqdm


def logprint(log):
    #print(log, end='')
    with open(logs_file, 'a') as f:
        f.write(log)


def train_model(model, optimizer, dataloader, args):
    model.train()
    n_batches = len(dataloader)

    avg_rec_loss = 0.
    avg_cls_loss = 0.
    avg_vq_loss = 0.
    avg_commit_loss = 0.

    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(tqdm(trainDataloader, ascii=True)):

        optimizer.zero_grad()

        if args.model=="vq":

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

            if args.use_wandb:
                wandb.log({
                    "Reconstruction Loss": avg_rec_loss,
                    "Classification Loss": avg_cls_loss,
                    "Total Loss": loss.item(),
                    "VQ Loss": avg_vq_loss,
                    "Commit Loss": avg_commit_loss
                })

            logprint(f'{idx + 1}/{n_batches} ')
            logprint(f'rec_loss: {avg_rec_loss / (idx + 1) :.6f} ')
            logprint(f'cls_loss: {avg_cls_loss / (idx + 1) :.6f} ')
            logprint(f'vq_loss: {avg_vq_loss / (idx + 1) :.6f} ')
            logprint(f'commit_loss: {avg_commit_loss / (idx + 1) :.6f}\n')

        elif args.model == "transformers" or args.model == "pointnet":
            rec_loss, classifier_loss = model(pc_corrupt.cuda(), pc_sampled.cuda(), y_corrupt.cuda())
            loss = rec_loss + args.classifier_coeff * classifier_loss

            avg_rec_loss += rec_loss.item()
            avg_cls_loss += classifier_loss.item()

            if args.use_wandb:
                wandb.log({
                    "Reconstruction Loss": rec_loss.item(),
                    "Classification Loss": classifier_loss.item(),
                    "Total Loss": loss.item()
                })

            if (idx%20):
                logprint(f'{idx + 1}/{n_batches} ')
                logprint(f'rec_loss: {avg_rec_loss / (idx + 1) :.6f} ')
                logprint(f'cls_loss: {avg_cls_loss / (idx + 1) :.6f} ')

        loss.backward()
        optimizer.step()

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


def validate(training_model, val_model):
    sd = training_model.state_dict()
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

    svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_labels.data.cpu().numpy(),
                           test_features.data.cpu().numpy(), test_labels.data.cpu().numpy())

    return svm_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_dir', type=str, default='models/pretrain_logs/')
    parser.add_argument('--dataroot', type=str, default="./data/ShapeNet55/ShapeNet55/")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--initial_epochs', type=int, default=10)
    parser.add_argument('--step_per_update', type=int, default=1)
    parser.add_argument('--classifier_coeff', type=float, default=0.25)
    parser.add_argument('--commit_coef', type=float, default=0.25)
    parser.add_argument('--emb_coef', type=float, default=0.25)
    parser.add_argument('--model', type=str, required=True) # Available options: vq, pointnet, transformers
    parser.add_argument('--use_wandb', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        os.mkdir(args.logs_dir)

    logs_file = os.path.join(args.logs_dir, 'logs.txt')

    assert args.model in ["vq", "pointnet", "transformers"], "Model not implemented"

    if args.use_wandb:
        what_is_running = input("Name this experiment for wandb log: ")
        wandb.init(name=what_is_running, project="674", entity="682_dbdc_dm")
        wandb.config.update(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')

    train_transforms = transforms.Compose(
        [
            PointcloudScaleAndTranslate(),
        ]
    )

    dataset = ShapeNet('train', train_transforms, dataroot= args.dataroot)
    trainDataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    dataset = ShapeNet('test', dataroot=args.dataroot)
    testDataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    if args.model == "vq":
        from models.vq_vae import VASP as SERP_Point_vq

        model = VASP().to(device)
        val_model = VASP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay,
                                      )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=args.epochs,
                                                               eta_min=1e-6)
    elif args.model == "transformers":
        from models.transformers import Point_SERP as SERP_Point_trf

        model = Point_SERP().to(device)
        val_model = Point_SERP(encoder_only=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay,
                                      )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=args.epochs,
                                                               eta_min=1e-6)
    elif args.model == "pointnet":
        from models.pointnet import SerpPointNet as SERP_Point_PointNet

        model = SERP_Point_PointNet().to(device)
        val_model = SERP_Point_PointNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    else:
        raise NotImplementedError

    max_svm_acc = 0.

    for epoch in range(args.epochs):

        avg_rec_loss, avg_cls_loss = train_model(model, optimizer, trainDataloader, args)
        scheduler.step()

        logprint(f'epoch:{epoch+1}/{args.epochs} rec_loss: {avg_rec_loss :.4f} cls:{avg_cls_loss :.4f}\n')


        svm_acc = validate(training_model=model, val_model=val_model)

        logprint(f'epoch:{epoch+1}/{args.epochs} svm_acc:{svm_acc}\n')

        if args.use_wandb:
            wandb.log({
                "Average Reconstruction Loss per epoch": avg_rec_loss,
                "Average Classification Loss per epoch": avg_cls_loss,
                "SVM accuracy": svm_acc
            })

        if max_svm_acc < svm_acc:

            max_svm_acc = svm_acc

            ckpt = {
                'best_model_wts' : model.state_dict(),
                'max_svm_acc' : max_svm_acc
            }

            path = os.path.join(args.logs_dir, 'model.pth')
            torch.save(ckpt, path)
