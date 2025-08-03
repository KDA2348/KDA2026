# Knowledge Distillation Experiments

## Prerequisites
```bash
conda create --name kd python=3.8 -y
conda activate kd
pip install -r requirements.txt
```

## Training
```bash
Execute the main script to start the MLKD_KDA process:


    python tools/train.py --cfg configs/cifar100/mlkd_kda/resnet32x4_resnet8x4.yaml 
    python tools/train.py --cfg configs/cifar100/mlkd_kda/resnet32x4_wrn16-2.yaml 
    python tools/train.py --cfg configs/cifar100/mlkd_kda/wrn40-2_resnet8x4.yaml 
    python tools/train.py --cfg configs/cifar100/mlkd_kda/wrn40-2_wrn16-2.yaml 
    python tools/train.py --cfg configs/imagenet/r34_r18/mlkd_kda.yaml 

 Execute the main script to start the KD_KDA process:

    python tools/train.py --cfg configs/cifar100/kd_kda/resnet32x4_resnet8x4.yaml 
    python tools/train.py --cfg configs/cifar100/kd_kda/resnet32x4_wrn16-2.yaml
    python tools/train.py --cfg configs/cifar100/kd_kda/wrn40-2_resnet8x4.yaml
    python tools/train.py --cfg configs/cifar100/kd_kda/wrn40-2_wrn16-2.yaml 
    python tools/train.py --cfg configs/imagenet/r34_r18/kd_kda.yaml
```

## Training teacher
```bash
python tools/train_teacher.py --cfg configs/teacher_resnet32x4.yaml
```

## References
- [Logit Standardization KD](https://deepwiki.com/sunshangquan/logit-standardization-KD)
