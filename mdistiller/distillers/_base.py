import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def extract_layer_weights(model: torch.nn.Module, layer_name_substring: str) -> Dict[str, Any]:
    """
    Extract weights from layers containing the given substring
    """
    weights = {}
    for name, param in model.named_parameters():
        if layer_name_substring in name and 'weight' in name:
            weights[name] = param.detach().cpu().numpy()
    return weights


def visualize_weights(weights: dict, model_label: str) -> None:
    for name, w in weights.items():
        plt.figure(figsize=(6, 4))

        if w.ndim == 2:
            sns.heatmap(w, cmap='viridis')
            plt.title(f"{model_label} | {name} (Heatmap)")
        else:
            plt.hist(w.flatten(), bins=50)
            plt.title(f"{model_label} | {name} (Histogram)")

        plt.xlabel("Weight Value")
        plt.ylabel("Frequency" if w.ndim != 2 else "Output Neurons")
        plt.tight_layout()
        plt.show()


class Distiller(nn.Module):
    student: nn.Module
    teacher: nn.Module

    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
