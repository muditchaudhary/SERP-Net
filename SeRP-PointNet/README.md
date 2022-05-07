# SeRP-PointNet

Download the data, saved models and install the required libraries mentioned in the main README.md in the home directory. 

## How to pre-train   
To pre-train for delta-learning run:
```
python pretrain.py --epochs <num_epochs> --model pointnet --rec_loss mse --learn_diff --dataroot ../data
```  
To pre-train for cdl2-learning run:
```
python pretrain.py --epochs <num_epochs> --model pointnet --rec_loss cdl2 --dataroot ../data
```

## How to fine-tune  
To fine-tune on ModelNet40:  
```
python finetune.py --epochs 50 --model pointnet --backbone_ckpt <pre-trained model ckpt or None> --dataset modelnet --dataroot ../data
```  
To fine-tune on ShapeNet:  
```
python finetune.py --epochs 50 --model pointnet --backbone_ckpt <pre-trained model ckpt or None> --dataset shapenet --dataroot ../data
```

## How to evaluate
```
python eval.py --model_ckpt <finetuned model path> --dataset <shapenet or modelnet> --dataroot ../data
```
