"""
Manually aligned layers in MLKD distillation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from typing import Callable
import numpy as np
from typing import Dict, Any, OrderedDict, Tuple
from yacs.config import CfgNode as ConfigNode

from ._base import Distiller
from ._base import extract_layer_weights, visualize_weights


def write_layer_output_to_activation_dict(name: str, activations_dict: Dict[str, Tensor]) -> Callable[[Module, Tensor, Tensor], None]:
    """
    Writes the output of a layer to the activations dictionary.

    Args:
        name (str): The name of the layer.
        activations_dict (Dict[str, Tensor]): The dictionary to store activations.

    Returns:
        Callable[[Module, Tensor, Tensor], None]: A hook function that writes the output of a layer to the activations dictionary.

    >>> import torch
    >>> import torch.nn as nn
    >>> from torch import Tensor
    >>> from typing import Dict
    >>> activations: Dict[str, Tensor] = {}
    >>> layer = nn.ReLU()
    >>> hook = write_layer_output_to_activation_dict("relu", activations)
    >>> _ = layer.register_forward_hook(hook)
    >>> x = torch.tensor([[-1.0, 0.0, 1.0]])
    >>> _ = layer(x)
    >>> torch.equal(activations["relu"], torch.tensor([[0.0, 0.0, 1.0]]))
    True
    """
    def hook(model: Module, input: Tensor, output: Tensor) -> None:
        activations_dict[name] = output
    return hook


def normalize(logit):
    eps = 1e-7
    mean = torch.mean(logit, dim=-1, keepdim=True)
    var = torch.var(logit, dim=-1, unbiased=False, keepdim=True)
    return (logit - mean) / torch.sqrt(var + eps)


def kd_loss(
        logits_student_in: Tensor,
        logits_teacher_in: Tensor,
        temperature: float,
        reduce: bool = True,
        logit_stand: bool = False) -> Tensor:
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def layer_kl_loss(
        s_feat: Tensor,
        t_feat: Tensor,
        temperature: float,) -> float:
    """
    Compute the KL divergence loss.

    Args:
        s_feat (Tensor): Student feature tensor.
        t_feat (Tensor): Teacher feature tensor.
        temperature (float): Temperature parameter.

    Returns:
        float: The computed KL divergence loss for each layer.
    """
    log_pred_student = F.log_softmax(s_feat / temperature, dim=1)
    pred_teacher = F.softmax(t_feat / temperature, dim=1)
    loss_kl: Tensor = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (temperature ** 2)
    return loss_kl.item()


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss

def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, use_cuda: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    lam = lam.reshape(-1, 1, 1, 1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class AlignedMLKD(Distiller):
    teacher_weights: Dict[str, Any] = {}
    student_weights: Dict[str, Any] = {}

    layers_losses: Dict[str, float] = {}
    proj: Dict[str, float] = {}
    # layer_loss = 0.0
    proj_linear_layer: nn.Linear

    def __init__(self, student: torch.nn.Module, teacher: torch.nn.Module, cfg: ConfigNode) -> None:
        """
        `Distiller` class will initialize the student and teacher models.

        Args:
            student (torch.nn.Module): Student model.
            teacher (torch.nn.Module): Teacher model.
            cfg (ConfigNode): Training configuration.
        """
        # Set a higher timeout for slow repr warnings during debugging
        os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1.5'

        assert isinstance(teacher, torch.nn.Module), "Teacher must inherit from torch.nn.Module"
        assert isinstance(student, torch.nn.Module), "Student must inherit from torch.nn.Module"

        super().__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        # self.align_layers = 0.0001
        # self.proj_linear_layer = nn.Linear(9, 9).to('cuda')
        # self.teacher_weights = self.teacher.state_dict()
        # self.student_weights = self.student.state_dict()
        # self.teacher_weights = { k:v for k, v in self.teacher.named_parameters() if 'conv' in k and len(v.shape) == 4}
        # self.student_weights = {k:v for k, v in self.student.named_parameters() if 'conv' in k and len(v.shape) == 4}
        # self.student_layer_head: int = 0
        # self.alignment_map = {}
        # self.maping_max: int = 0 #
        # self.max_maping = -(-len(self.teacher_weights) // len(self.student_weights)) 
        # for i in range(len(self.teacher_weights) - 1):
        #     key_teacher = list(self.teacher_weights.keys())[i]
        #     curr_student_index = min(self.student_layer_head, len(self.student_weights) - 1)
        #     key_student_curr = list(self.student_weights.keys())[curr_student_index]
        #     next_student_index = self.student_layer_head + 1
        #     # next_student_index = min(next_student_index, len(self.student_weights) - 1)

        #     key_student_next = list(self.student_weights.keys())[next_student_index]
        #     shape = self.teacher_weights[key_teacher].cuda().shape

        #     student_layer_weights_current = self.student_weights[key_student_curr].cuda().view(-1, shape[-1] * shape[-2])
        #     student_layers_weights_next = self.student_weights[key_student_next].cuda().view(-1, shape[-1] * shape[-2])

        #     teacher_layer_weights = self.teacher_weights[key_teacher].cuda().view(-1, shape[-1] * shape[-2])
        #     result: str = self.get_closest_aligned_layer(student_layer_weights_current, student_layers_weights_next, teacher_layer_weights)

        #     self.maping_max = 0
        #     if result == "current":
        #         if self.maping_max <= self.max_maping:
        #             self.maping_max += 1
        #             self.alignment_map[key_teacher] = key_student_curr
        #         else:
        #             self.maping_max = 0
        #             self.alignment_map[key_teacher] = key_student_next
        #             self.student_layer_head += 1
        #     else:
        #         self.maping_max = 0
        #         self.alignment_map[key_teacher] = key_student_next
        #         self.student_layer_head += 1

        # To print the named parameters of models:
        # for name, param in self.teacher.named_parameters():
        #     print(f"Teacher - {name}: {param.data}")
        # for name, param in self.student.named_parameters():
        #     print(f"Student - {name}: {param.data}")

        # To copy the weights from teacher to student:
        # self.student.load_state_dict(self.teacher.state_dict(), strict=False)

        # To print all the layers of each model:
        # print("Teacher Model Layers:")
        # for name, param in self.teacher_weights.items():
        #     print(f"{name} => shape: {tuple(param.shape)}")
        # print("\nStudent Model Layers:")
        # for name, param in self.student_weights.items():
        #     print(f"{name} => shape: {tuple(param.shape)}")

        # Only print the weights, ignoring the buffers
        # for name, param in self.teacher.named_parameters():
        #     if param.requires_grad:
        #         print(f"Teacher - {name}")
        # for name, param in self.student.named_parameters():
        #     if param.requires_grad:
        #         print(f"Student - {name}")

        # print("\nLayer-by-layer comparison:")
        # for name in self.teacher_weights:
        #     if name in self.student_weights:
        #         t_shape = tuple(self.teacher_weights[name].shape)
        #         s_shape = tuple(self.student_weights[name].shape)
        #         match = t_shape == s_shape
        #         if match:
        #             print(f"{name}: teacher {t_shape}, student {s_shape} => MATCH")
        #     # else:
        #     #     print(f"{name} present in teacher but missing in student")

        # To access the weights of a specific layer:
        # layer_name = 'layer_name_here'
        # if layer_name in self.teacher_weights:
        #     layer_weights = self.teacher_weights[layer_name]
        #     print(f"Teacher {layer_name} weights: {layer_weights}")

        # assert isinstance(self.teacher_weights["layer1.0.conv1.weight"], torch.Tensor), "Teacher weights should be a tensor"

        # student_biases: Dict[str, torch.Tensor] = {}
        # for name, param in self.student.named_parameters():
        #     if 'bias' in name:
        #         student_biases[name] = param.data.clone()

        print("Teacher and Student model weights initialized.")
        return

    def visualize_model_weights(self, layer_name_filter: str = '') -> None:
        """
        Visualizes the weights of the student and teacher models.

        Args:
            layer_name_filter (str, optional): Filter for layer names. Defaults to ''.
        """
        student_weights = extract_layer_weights(self.student, layer_name_filter)
        teacher_weights = extract_layer_weights(self.teacher, layer_name_filter)

        visualize_weights(student_weights, "Student Model")
        visualize_weights(teacher_weights, "Teacher Model")
        return

    
    def mean_max_cosine_similarity(self, A: torch.Tensor, B: torch.Tensor) -> Tensor:
        A_norm = F.normalize(A, dim=1) 
        B_norm = F.normalize(B, dim=1)  
        cosine_sim = A_norm @ B_norm.T 
        cosine_dist = 1 - cosine_sim
        return cosine_dist.mean()

 
    def get_closest_aligned_layer(self, current_student_layer: Tensor, next_student_layer: Tensor, teacher_layer: Tensor) -> str:
        diff_curr: float = self.mean_max_cosine_similarity(current_student_layer, teacher_layer)
        diff_next: float = self.mean_max_cosine_similarity(next_student_layer, teacher_layer)
        return "current" if diff_next > diff_curr else "next"

    def forward_train(  # type: ignore
        self,
        image_weak: torch.Tensor,
        image_strong: torch.Tensor,
        target: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs a forward pass during training using weak and strong augmentations.

        Args:
            image_weak (torch.Tensor): Input image with weak augmentation.
            image_strong (torch.Tensor): Input image with strong augmentation.
            target (torch.Tensor): Ground truth labels.

        Returns:
            Tuple: Total loss and a dictionary of intermediate tensors.
        """
        # self.teacher_weights = {
        #     k:v for k, v in self.teacher.named_parameters() if 'conv' in k and len(v.shape) == 4
        # }
        # # self.student_weights = {
        # #     k:v for k, v in self.student.named_parameters() if 'conv' in k
        # # }
        # self.student_weights = {
        #     k:v for k, v in self.student.named_parameters() if 'conv' in k and len(v.shape) == 4
        # }
        # print the shape of the student weights
        # for name, param in self.student_weights.items():
        #     print(f"{name} => shape: {tuple(param.shape)}")
        # num_layers_teacher = len(self.teacher_weights)
        # num_layers_student = len(self.student_weights)
        # student_layer_head: int = 0
        # layer_loss: Tensor = torch.tensor(0.0, device='cuda')
        # for i in range(num_layers_teacher - 1):
        #     key_teacher = list(self.teacher_weights.keys())[i]
        #     curr_student_index = min(student_layer_head, num_layers_student - 1)
        #     key_student_curr = list(self.student_weights.keys())[curr_student_index]
        #     next_student_index = student_layer_head + 1
        #     next_student_index = min(next_student_index, num_layers_student - 1)

        #     key_student_next = list(self.student_weights.keys())[next_student_index]
        #     shape = self.teacher_weights[key_teacher].cuda().shape

        #     student_layer_weights_current = self.student_weights[key_student_curr].cuda().view(-1, shape[-1] * shape[-2])
        #     student_layers_weights_next = self.student_weights[key_student_next].cuda().view(-1, shape[-1] * shape[-2])

        #     teacher_layer_weights = self.teacher_weights[key_teacher].cuda().view(-1, shape[-1] * shape[-2])
        #     result: str = self.get_closest_aligned_layer(student_layer_weights_current, student_layers_weights_next, teacher_layer_weights)

        #     if result == "current":
        #         # Find the shape of the 2 aligned layers
        #         # Construct the nn.Linear layer
        #         student_layer_proj = self.proj_linear_layer(student_layer_weights_current)
        #         alignment_map[key_teacher] = key_student_curr
        #         # layer_loss += F.kl_div(student_layer_proj, teacher_layer_weights, reduction='batchmean')
        #     else:
        #         alignment_map[key_teacher] = key_student_next
        #         student_layer_proj = self.proj_linear_layer(student_layers_weights_next)
        #         # layer_loss += F.kl_div(student_layer_proj, teacher_layer_weights, reduction='batchmean')
        #         # print(f"{result}")
        #         # storage_key = f"layer_{i}_{j}_diff"
        #         # self.layers_losses[key] = layer_diff.item()
        #         student_layer_head += 1
        # for key_teacher, key_student in self.alignment_map.items():
        #     shape = self.teacher_weights[key_teacher].cuda().shape
        #     student_weights = self.student_weights[key_student].cuda().view(-1, shape[-1] * shape[-2])
        #     teacher_weights = self.teacher_weights[key_teacher].cuda().view(-1, shape[-1] * shape[-2])

        #     student_proj = self.proj_linear_layer(student_weights)
        #     layer_loss += self.mean_max_cosine_similarity(student_proj, teacher_weights)
        #layer_loss += self.mean_max_cosine_similarity(student_layer_proj, teacher_layer_weights)
        # second_key: str = list(self.teacher_weights.keys())[1]
        # init_teacher_weights = self.teacher_weights[second_key].cuda()
        # init_student_weights = self.student_weights[list(self.teacher_weights.keys())[0]].cuda()
        # student_weights_proj = self.proj_linear_layer(init_student_weights)
        # kl_student_teacher_weights = F.kl_div(student_weights_proj, init_teacher_weights, reduction='batchmean')

        # Use manual layers alignment for the teacher and student models
        # student_layers: list[str] = ["conv1", "bn1", "layer2", "fc"]
        # teacher_layers: list[str] = ["conv1", "bn1", "layer1", "fc"]
        # student_layers: list[str] = ["conv1", "bn1", "fc"]
        # teacher_layers: list[str] = ["conv1", "bn1", "fc"]

        # Activation storage
        # student_activations: Dict[str, torch.Tensor] = {}
        # teacher_activations: Dict[str, torch.Tensor] = {}

        # Register hooks
        # for s_layer_name, t_layer_name in zip(student_layers, teacher_layers):
        #     getattr(self.student, s_layer_name).register_forward_hook(
        #         write_layer_output_to_activation_dict(s_layer_name, student_activations)
        #     )
        #     getattr(self.teacher, t_layer_name).register_forward_hook(
        #         write_layer_output_to_activation_dict(t_layer_name, teacher_activations)
        #     )

        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)

        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        # batch_size, class_num = logits_student_strong.shape

        # all_distill_layers_losses: float = 0.0
        # s_feat: Tensor
        # t_feat: Tensor
        # Compute layers losses
        # for s_key, t_key in zip(student_layers, teacher_layers):
        #     s_feat = student_activations[s_key]
        #     t_feat = teacher_activations[t_key]

        #     # Resize teacher features if channels differ
        #     if hasattr(s_feat, "shape") and hasattr(t_feat, "shape") and s_feat.shape[1] != t_feat.shape[1]:
        #         t_feat_broadcast_shape: Tuple[int, int] = tuple(s_feat.shape[2:])  # type: ignore
        #         t_feat = torch.nn.functional.adaptive_avg_pool2d(t_feat, t_feat_broadcast_shape)
        #         t_feat = t_feat[:, :s_feat.shape[1], :, :]  # truncate if necessary

        #     # Save layers losses to a dictionary
        #     # self.layers_losses[s_key] = F.mse_loss(input=s_feat, target=t_feat).item()
        #     # all_distill_layers_losses += self.layers_losses[s_key]
        #     # Compute kl loss and ce loss for layers
        #     layer_loss_kl: float = layer_kl_loss(
        #         s_feat=s_feat,
        #         t_feat=t_feat,
        #         temperature=self.temperature,
        #     )
        #     layer_loss_ce: float = (F.cross_entropy(input=logits_student_weak, target=target) + F.cross_entropy(input=logits_student_strong, target=target)).item()
        #     self.layers_losses[s_key] = 0.5 * layer_loss_kl + 0.5 * layer_loss_ce

        #     if t_feat.shape == s_feat.shape:
        #         student_flat = s_feat.flatten()
        #         teacher_flat = t_feat.flatten()
        #         self.proj[f"{s_key}_proj_cosine"] = F.cosine_similarity(student_flat, teacher_flat, dim=0).item()
        #         self.proj[f"{s_key}_proj_mse"] = F.mse_loss(student_flat, teacher_flat).item()
        #         self.proj[f"{s_key}_proj_norm"] = torch.norm(student_flat - teacher_flat).item()
        #         self.proj[f"{s_key}_proj_corr"] = torch.stack([student_flat, teacher_flat])[0, 1].item()
        #     # TODO: add dynamic channels handling

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, _ = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh: np.float64 = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask: torch.Tensor = confidence.le(float(conf_thresh)).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh: np.float64 = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(float(class_confidence_thresh)).bool()

        # losses
        loss_ce: Tensor = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_kd_weak: Tensor = self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            logit_stand=self.logit_stand,
        ) * mask).mean())

        loss_kd_strong = self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            logit_stand=self.logit_stand,
        )

        loss_cc_weak = self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * class_conf_mask).mean())
        loss_bc_weak = self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * mask).mean())
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak,
            # "loss_layer": self.align_layers * layer_loss
        }
        # print(f"Layer loss: {layer_loss}")
        return logits_student_weak, losses_dict
