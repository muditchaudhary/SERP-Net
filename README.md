# SERP-Net

## Setup
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  
cd extensions/chamfer_dist  
python setup.py install --user  
cd ../../  
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install tqdm wandb pandas opencv-python matplotlib timm
```

## Links to download data and trained models

Data link: https://drive.google.com/file/d/18fWjXj2Io6kcHOe9DKnInFliAvveKJMA/view?usp=sharing

SeRP-PointNet Saved Models: https://drive.google.com/file/d/1bn8XhYt4e6UCklfsQv-Tikk_kDPOwswV/view?usp=sharing

SeRP-Transformer Saved Models: https://drive.google.com/file/d/12LJmrf5AyBxlqZWx6Bn4UtXWUIJ66rPL/view?usp=sharing

## The final file structure should be:
```
|  
|-- data/
|         |--ModelNet/
|         |     |--modelnet40_train_8192pts_fps.dat
|         |     |--modelnet40_test_8192pts_fps.dat
|         |  
|         |--ShapeNet55/
|         |     |--ShapeNet55/
|         |     |       |--label_ids.pth
|         |     |       |--train_split.csv
|         |     |       |--val_split.csv
|         |     |       |--shapenet_pc/
|         |     |       |       |--pc_1.npy
|         |     |       |       |--pc_2.npy
|         |     |       |       |--.....npy
|         |     |       |       |--.....npy
|         |     |       |       |--pc_N.npy
|         |         
|
|-- SeRP-PointNet/
|         |
|         |--saved_models/
|         | ... <remaining files>
|
|-- SeRP-transformer
|         |--models/
|         |     |--pre-trained/  # pre-trained models on ShapeNet-55
|         |     |   |--tr_serp/     # pre-trained SeRP-Transformer
|         |     |   |--tr_vasp/     # pre-trained VASP-Transformer
|         |     |   
|         |     |--fine-tuned/  # fine-tuned models on classification tasks
|         |     |   |--m40_no_pretrain/  # encoder trained from scratch on ModelNet40 dataset
|         |     |   |--m40_tr_serp/  # Pre-trained SeRP encoder on ModelNet40 dataset
|         |     |   |--m40_tr_serp/  # Pre-trained VASP encoder on ModelNet40 dataset
|         |     |   |--sh_tr_nopretrain/  # encoder trained from scratch on ShapeNet classification dataset
|         |     |   |--sh_tr_serp/  # Pre-trained SeRP encoder on ShapeNet55 dataset
|         |     |   |--sh_tr_vasp/  # Pre-trained VASP encoder on ShapeNet55 dataset
|         |     |   
|         |     |--pretrain/  # by default to store pretrained models while pre-training
|         |     |   --model.pth  # this will be written here
|         |     |   --logs.txt  # this will be written here
|         |     |   
|         |     |--finetune/  # by default to store classification models while finetuning
|         |     |   --model.pth  # this will be written here
|         |     |   --logs.txt  # this will be written here
|
|-- extensions/
```

## How to run  
Instructions on how to run `SeRP-PointNet` and `SeRP-transformer` are provided in the `README.md` in the respective directories.

## Source code citations and Acknowledgments
Some parts of `SeRP-PointNet/pointnet.py` were sourced from https://github.com/fxia22/pointnet.pytorch

Transformer model was adapted from [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)

Vector-Quantization operations and gradient operations were adapted from [VQ-VAE](https://github.com/jaywalnut310/Vector-Quantized-Autoencoders)

Processed datasets for ShapeNet55 and ModelNet40 are taken from [Point-BERT](https://github.com/lulutang0608/Point-BERT/tree/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52)

## Contributions
Mudit Chaudhary: 
```
SeRP-PoinNet/models/pointnet.py
SeRP-PointNet/eval.py  
SeRP-PointNet/finetune.py <with Siddhant Garg>
SeRP-PointNet/pretrain.py <with Siddhant Garg>
```  
Siddhant Garg:
```
Implemented by Siddhant Garg (full script)
  |--SeRP-transformer/data_utils.py
  |--SeRP-transformer/evaluate_classification.py
  |--SeRP-transformer/finetune.py
  |--SeRP-transformer/generate_vasp.py
  |--SeRP-transformer/plot_tsne.py
  |--SeRP-transformer/reconstruct.py
  |--SeRP-transformer/utils.py

Implemented by Siddhant Garg (partial scripts)
  |--SeRP-transformer/serp_transformer.py (Implemented class Point_SERP from lines 256-396)
  |--SeRP-transformer/vq_vae.py (Implemented the class VASP from lines 117-277)
 
Implemented jointly with Mudit Chaudhary
  |--SeRP-transformer/pretrain.py 
  |--SeRP-transformer/finetune.py
```
