# Knowledge Distillation Experiments

## Prerequisites
```bash
conda create --name kd python=3.8 -y
conda activate kd
pip install -r requirements.txt
python scripts/download_checkpoints.py
```

## Training
```bash
cd logit-standardization-KD
python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9
```

## Training teacher
```bash
cd logit-standardization-KD
python tools/train_teacher.py --cfg configs/teacher_resnet32x4.yaml
```

## References
- [Logit Standardization KD](https://deepwiki.com/sunshangquan/logit-standardization-KD)
