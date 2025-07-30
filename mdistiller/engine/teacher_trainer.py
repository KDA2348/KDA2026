"""
Trainer module for teacher model in the knowledge distillation process.

This class handles training, evaluation, logging, and checkpointing.
"""

import os
import time
from collections import OrderedDict
from typing import Any
import torch.nn.functional as F

import torch
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .utils import (
    AverageMeter,
    accuracy,
    validate_model,
    adjust_learning_rate,
    save_checkpoint,
    log_msg,
)


class TeacherTrainer:
    """
    Manages the training of the teacher model for the distillation pipeline.
    """

    def __init__(
        self,
        experiment_name,
        model, train_loader: DataLoader, val_loader: DataLoader, cfg: Any
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self._init_optimizer(cfg)
        self.best_acc = -1.0
        self.start_epoch = 0

        # Setup logging directory and writer
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.tf_writer = SummaryWriter(
            os.path.join(self.log_path, "train.events"))

    def _init_optimizer(self, cfg: Any) -> SGD:
        """
        Initializes the SGD optimizer using configuration.
        """
        return SGD(
            self.model.parameters(),
            lr=cfg.SOLVER.LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    def _log_metrics(self, epoch: int, lr: float, metrics: Any) -> None:
        """
        Logs training and validation metrics.
        """
        for key, value in metrics.items():
            self.tf_writer.add_scalar(key, value, epoch)
        self.tf_writer.flush()

        if metrics.get("test_acc", -1) > self.best_acc:
            self.best_acc = metrics["test_acc"]

        log_lines = [
            "-" * 25,
            f"Epoch: {epoch}",
            f"Learning Rate: {lr:.6f}",
        ] + [f"{k}: {v:.2f}" for k, v in metrics.items()] + ["-" * 25]

        with open(os.path.join(self.log_path, "worklog.txt"), "a", encoding="utf-8") as f:
            f.write(os.linesep.join(log_lines) + os.linesep)

    def train(self):
        """
        Executes the training loop across configured epochs.
        """
        epoch = 1
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self._train_one_epoch(epoch)
            epoch += 1

        print(log_msg(f"Best accuracy: {self.best_acc:.4f}", "EVAL"))

        with open(os.path.join(self.log_path, "worklog.txt"), "a", encoding='utf-8') as f:
            f.write(f"best_acc\t{self.best_acc:.4f}\n")

    def _train_one_epoch(self, epoch: int):
        """
        Trains the teacher model for a single epoch.
        """
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter: int = len(self.train_loader)
        pbar = tqdm(range(num_iter), desc="TRAIN")
        self.model.train()

        for _, data in enumerate(self.train_loader):
            msg: str = self._train_batch(data, epoch, meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # Validation
        test_acc, test_acc_top5, test_loss = validate_model(self.val_loader, self.model)

        metrics = OrderedDict({
            "train_acc": meters["top1"].avg,
            "train_loss": meters["losses"].avg,
            "test_acc": test_acc,
            "test_acc_top5": test_acc_top5,
            "test_loss": test_loss,
        })

        self._log_metrics(epoch, lr, metrics)

        # Checkpointing
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }

        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        #print(os.path.join(self.log_path, "latest"))
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(state, os.path.join(
                self.log_path, f"epoch_{epoch}"))
        if test_acc >= self.best_acc:
             save_checkpoint(state, os.path.join(self.log_path, "best"))

    def _train_batch(self, data, epoch, meters) -> str:
        """
        Processes a single batch during training.
        """
        self.optimizer.zero_grad()
        start_time = time.time()

        images, targets, _ = data
        meters["data_time"].update(time.time() - start_time)

        images, targets = images.float().cuda(
            non_blocking=True), targets.cuda(non_blocking=True)

        outputs, _ = self.model(images)
        assert isinstance(outputs, torch.Tensor)

        loss = F.cross_entropy(outputs, targets)

        # Backward pass and optimizer step
        loss.backward()
        self.optimizer.step()

        # Timing and metrics
        meters["training_time"].update(time.time() - start_time)

        batch_size = images.size(0)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        meters["losses"].update(loss.item(), batch_size)
        meters["top1"].update(acc1[0], batch_size)
        meters["top5"].update(acc5[0], batch_size)

        return (
            f"Epoch: {epoch} | Data Time: {meters['data_time'].avg:.3f}s | "
            f"Train Time: {meters['training_time'].avg:.3f}s | Loss: {meters['losses'].avg:.4f} | "
            f"Top-1 Acc: {meters['top1'].avg:.2f}% | Top-5 Acc: {meters['top5'].avg:.2f}%"
        )
