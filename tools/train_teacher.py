"""
This module is responsible for training the teacher model in the knowledge distillation process.
It handles the configuration, data loading, and training loop.
"""

import sys
import os
import argparse
import time
import torch
from torch.backends import cudnn
cudnn.benchmark = True

# Add the project root and mdistiller directory to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, "mdistiller"))

from mdistiller.engine.cfg import show_cfg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.dataset import get_dataset
from mdistiller.models import cifar_model_dict
from mdistiller.engine.teacher_trainer import TeacherTrainer

def main(cfg, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        additional_tags: list[str] = [
            "{}:{}".format(k, v)
            for k, v in zip(opts[::2], opts[1::2])
        ]
        tags += additional_tags
        experiment_name += ",".join(additional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)

    show_cfg(cfg)

    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    if cfg.DISTILLER.TYPE == "NONE":
        print("Training the teacher model only to reproduce check points")

    network_constructor_fn, _ = cifar_model_dict[cfg.DISTILLER.TEACHER]

    model_teacher = network_constructor_fn(num_classes=num_classes)
    model_teacher = torch.nn.DataParallel(model_teacher.cuda())
    total_params = sum(p.numel() for p in model_teacher.parameters())
    trainable_params = sum(p.numel() for p in model_teacher.parameters() if p.requires_grad)
    print(f"🧮 Total parameters: {total_params:,}")
    print(f"🧮 Trainable parameters: {trainable_params:,}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # make sure all GPU memory allocations are synchronized
    mem_alloc_before = torch.cuda.memory_allocated() / 1024**2
    mem_reserved_before = torch.cuda.memory_reserved() / 1024**2
    # train
    start_time = time.time()
    trainer: TeacherTrainer = TeacherTrainer(
        cfg.EXPERIMENT.NAME, model_teacher, train_loader, val_loader, cfg)
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    torch.cuda.synchronize()
    mem_alloc_after = torch.cuda.memory_allocated() / 1024**2
    mem_reserved_after = torch.cuda.memory_reserved() / 1024**2
    print(f"🕒 Training time: {training_time:.2f} seconds")
    print(f"🖥️ GPU Memory Allocated (Before → After): {mem_alloc_before:.2f} MB → {mem_alloc_after:.2f} MB")
    print(f"🖥️ GPU Memory Reserved (Before → After): {mem_reserved_before:.2f} MB → {mem_reserved_after:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.opts)
