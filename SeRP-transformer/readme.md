# SeRP Transformer

### Pre-training

#### SeRP-Transformer

```python pretrain.py --logs_dir=models/pretrain/tr_serp --loss_type=cdl2 --batch_size=128```

#### VASP-Transformer

```python pretrain.py --logs_dir=models/pretrain/tr_vasp --batch_size=64  --use_vq=True```

### Fine-Tune on downstream task 

Training the transformer encoder without pretrained model

```
python finetune.py \
--logs_dir=models/finetune/m40_no_pretrain \
--epochs=300 \
--dataset=modelnet \
--learning_rate=0.0001 \
--weight_decay=0.001
```

Training the transformer encoder with pretrained model

```
# SeRP-Transformer
python finetune.py \
--logs_dir=models/finetune/m40_tr_serp \
--prev_ckpt=models/pretrain/tr_serp \
--epochs=300 \
--dataset=modelnet \
--learning_rate=0.0001 \
--weight_decay=0.001 \
--dataset=DATASET

# VASP-Transformer
python finetune.py \
--logs_dir=models/finetune/m40_tr_vasp \
--prev_ckpt=models/pretrain/tr_vasp \
--epochs=300 \
--dataset=modelnet \
--learning_rate=0.0001 \
--weight_decay=0.001 \
--dataset=DATASET

DATASET = {modelnet, shapenet}
```

#### Evaluating fine-tuned models

```
python evaluate_classification.py \
--finetuned_model=models/finetune/TRAINED_MODEL_PATH

TRAINED_MODEL_PATH = {}
```
