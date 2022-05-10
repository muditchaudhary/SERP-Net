
# SeRP Transformer

### Requirements

```PyTorch >= 1.7.0; python >= 3.7; CUDA >= 9.0; GCC >= 4.9; torchvision;```



### Pre-training on ShapeNet

#### SeRP-Transformer

```
python pretrain.py --logs_dir=<LOGS_DIR to store pretrained model and logs.txt file> --loss_type=cdl2 --batch_size=128
# model.pth and logs.txt file will be stored as checkpoints in the LOGS_DIR provided
```

#### VASP-Transformer

```
python pretrain.py --logs_dir=<LOGS_DIR to store pretrained model and logs.txt file> --batch_size=64  --use_vq=True
# model.pth and logs.txt file will be stored as checkpoints in the LOGS_DIR provided
```

### Fine-Tune on downstream task 

Training the transformer encoder without pretrained model

```
# In the this command, logs_dir is the path to LOGS_DIR where you want to store the trained model. 
# Use dataset=shapenet to train on ShapeNet55 dataset or dataset=modelnet to train on ModelNet40 dataset

python finetune.py \
--logs_dir=LOGS_DIR \
--epochs=300 \
--dataset=modelnet \
--learning_rate=0.0001 \
--weight_decay=0.001

python finetune.py \
--logs_dir=LOGS_DIR \
--epochs=300 \
--dataset=shapenet \
--learning_rate=0.0001 \
--weight_decay=0.001
```

Training the transformer encoder with pretrained model

```
# SeRP-Transformer
python finetune.py \
--logs_dir=LOGS_DIR \ # directory in which the trained model is saved
--prev_ckpt=models/pre-trained/tr_serp/model.pth \ # path to the pre-trained transformer ckpt
--epochs=300 \
--dataset=modelnet \  # can use dataset=modelnet or dataset=shapenet
--learning_rate=0.0001 \
--weight_decay=0.001 

# VASP-Transformer
python finetune.py \
--logs_dir=LOGS_DIR \ # directory in which the trained model is saved
--prev_ckpt=models/pre-trained/tr_vasp/model.pth \  # path to the pre-trained transformer ckpt
--epochs=300 \
--dataset=modelnet \  # can use dataset=modelnet or dataset=shapenet
--learning_rate=0.0001 \
--weight_decay=0.001 \

DATASET = {modelnet, shapenet}
```

#### Evaluating fine-tuned models

```
# Transformer trained on ModelNet40 with random weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/m40_no_pretrain/model.pth

# Transformer trained on ShapeNet55 with random weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/sh_tr_nopretrain/model.pth

# SeRP-Transformer trained on ModelNet40 with pre-trained weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/m40_tr_serp/model.pth

# SeRP-Transformer trained on ShapeNet55 with pre-trained weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/sh_tr_serp/model.pth

# VASP trained on ModelNet40 with pre-trained weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/m40_tr_vasp/model.pth

# VASP trained on ShapeNet55 with pre-trained weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/sh_tr_vasp/model.pth
```
