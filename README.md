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
SERP-Net  
|  
|-- data/
|
|-- SeRP-PointNet/
|         |
|         |--saved_models/
|         | ... <remaining files>
|
|-- SeRP-transformers
|         |--saved_models/
|         | ... <remaining files>
|
|-- extensions/
```

## How to run  
Instructions on how to run `SeRP-PointNet` and `SeRP-transformer` are provided in the `README.md` in the respective directories.

## Source code citations
Some parts of `SeRP-PointNet/pointnet.py` were sourced from https://github.com/fxia22/pointnet.pytorch


## Contributions
Mudit Chaudhary: 
```
SeRP-PoinNet/models/pointnet.py
SeRP-PointNet/eval.py  
SeRP-PointNet/finetune.py <with Siddhant Garg>
SeRP-PointNet/pretrain.py <with Siddhant Garg>
```
