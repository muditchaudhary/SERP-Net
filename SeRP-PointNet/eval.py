from models.pointnet import SerpPointNet,SerpPointNetClassifier
import torch
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_utils import PointcloudScaleAndTranslate, ShapeNet, ModelNet40


def evaluate(model, dataloader,):
    model.eval()

    n_batches = len(dataloader)

    avg_acc = 0.

    for idx, (tax_id, pc_sampled, pc_corrupt, y_corrupt) in enumerate(tqdm(dataloader, ascii=True)):


        with torch.no_grad():
            logits = model(pc_sampled.cuda())

        pred = logits.argmax(-1).detach().cpu().numpy()
        labels = tax_id.detach().cpu().numpy()
        matches = (pred == labels).sum()
        avg_acc += matches

    avg_acc /= (n_batches * args.batch_size)

    return avg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='modelnet')
    parser.add_argument('--dataroot', type=str, default="../data")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose(
        [
            PointcloudScaleAndTranslate(),
        ]
    )

    if args.dataset == 'shapenet':

        dataset = ShapeNet('test', dataroot = args.dataroot)
        testDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = 55

    elif args.dataset == 'modelnet':
        dataset = ModelNet40('test', dataroot= args.dataroot)
        testDataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = 40

    backbone_model = SerpPointNet(finetuning=True)
    model = SerpPointNetClassifier(backbone=backbone_model, num_label=num_classes).to(device)

    checkpoint = torch.load(args.model_ckpt)
    model.load_state_dict(checkpoint['model_wts'])
    model.cuda()
    accuracy = evaluate(model, dataloader=testDataloader)
    print("Accuracy: ", accuracy)
