"""
KDA distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
from .MatchCov import build_auto_matching_pair


# def mean_pairwise_cosine_distance(A: torch.Tensor, B: torch.Tensor) -> Tensor:
#     """
#     Computes the mean pairwise cosine distance between two batches of vectors.

#     Each row in `A` and `B` represents a vector. This function normalizes both
#     inputs along dimension 1 (row-wise), computes all pairwise cosine similarities,
#     converts them to distances (1 - similarity), and returns the mean distance.

#     Args:
#         A (Tensor): A tensor of shape (N, D) representing N vectors of dimension D.
#         B (Tensor): A tensor of shape (M, D) representing M vectors of dimension D.

#     Returns:
#         Tensor: A scalar tensor representing the mean pairwise cosine distance
#                 between vectors in `A` and vectors in `B`.

#     Example:
#         >>> A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
#         >>> B = torch.tensor([[1.0, 1.0]])
#         >>> mean_pairwise_cosine_distance(A, B).round(decimals=4)
#         tensor(0.2929)

#         # cosine_similarity([1, 0], [1, 1]) = 1/sqrt(2) ≈ 0.7071, therefore distance ≈ 0.2929
#         # cosine_similarity([0, 1], [1, 1]) = 1/sqrt(2) ≈ 0.7071, therefore distance ≈ 0.2929
#         # Mean distance = (0.2929 + 0.2929) / 2 = 0.2929
#     """
#     A_norm = F.normalize(A, dim=1)
#     B_norm = F.normalize(B, dim=1)
#     cosine_sim = torch.matmul(A_norm, B_norm.T)
#     cosine_dist = 1 - cosine_sim
#     # cosine_sim = F.cosine_similarity(A, B, dim=1)  # [N]
#     # cosine_dist = 1 - cosine_sim
#     return cosine_dist.mean()

# def reduce_rows(x, target_rows):
#     if x.shape[0] == target_rows:
#         return x
#     elif x.shape[0] > target_rows:
#         # 直接均匀分块平均池化
#         factor = x.shape[0] // target_rows
#         x = x[:factor * target_rows].reshape(target_rows, factor, -1).mean(dim=1)
#         return x
#     else:
#         # 行数不足，repeat + 截断
#         repeat_times = (target_rows + x.shape[0] - 1) // x.shape[0]
#         x = x.repeat(repeat_times, 1)[:target_rows]
#         return x

def kd_loss(student_weight, teacher_weight, temperature, reduce=True):
    #student_weight= reduce_rows(student_weight,teacher_weight.shape[0])
    #teacher_weight= reduce_rows(teacher_weight,student_weight.shape[0])
    
    teacher_probs = F.softmax(teacher_weight, dim=1)
    student_log_probs = F.log_softmax(student_weight, dim=1)
    #student_weight,teacher_weight = align_tensor_batch_size(student_weight, teacher_weight)
    if reduce:
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(1)
    loss_kd *= temperature**2
    # print(loss_kd)
    return loss_kd


# def get_closest_aligned_layer(current_student_layer: Tensor, next_student_layer: Tensor, teacher_layer: Tensor) -> str:
#     diff_curr = mean_pairwise_cosine_distance(current_student_layer, teacher_layer)
#     diff_next = mean_pairwise_cosine_distance(next_student_layer, teacher_layer)
#     return "current" if diff_next > diff_curr else "next"


class KDA:
    def __init__(self, student: nn.Module, teacher: nn.Module, align_layers, Temrature, feature_dim: int = 9, device: str = 'cuda') -> None:
        self.device = torch.device(device)
        self.student = student
        self.teacher = teacher
        self.align_layers = align_layers
        self.Tem = Temrature
        print(self.align_layers, self.Tem)
        #T=[1,3,5,7,9,10]
        #al=[0.1,0.01,0.001,0.0001,0.00001,0.000001]
        self.proj_linear_layer = nn.Linear(feature_dim, feature_dim).to(self.device)

        self.teacher_weights = self._extract_conv_weights(self.teacher)
        self.student_weights = self._extract_conv_weights(self.student)

        self.student_layer_head = 0
        self.alignment_map: Dict[str, str] = {}
        self.max_mappings_per_layer = -(-len(self.teacher_weights) // len(self.student_weights))  # ceiling division

        self._build_alignment_map()

    def _extract_conv_weights(self, model: nn.Module) -> Dict[str, Tensor]:
        # for name, param in model.named_parameters():
            
        #     print(f"Layer: {name}, Shape: {param.shape}")
        
        # return {
        #     name: param for name, param in model.named_parameters()
        #     if 'conv' in name and param.ndim == 4
        # }
        return {
            name: param for name, param in model.named_parameters()
            if param.ndim == 4
        }

    # def _prepare_weights(self, weight: Tensor, shape: Tuple[int]) -> Tensor:
    #     return weight.to(self.device).view(-1, shape[-1] * shape[-2])
    def _prepare_weights(self, weight: Tensor, shape: Tuple[int]) -> Tensor:
        return weight.to(self.device).view(-1, shape[-1] * shape[-2])

    # def _build_alignment_map(self) -> None:
    #     mapping_count = 0
    #     teacher_keys = list(self.teacher_weights.keys())
    #     student_keys = list(self.student_weights.keys())

    #     for i in range(len(teacher_keys) - 1):
    #         key_teacher = teacher_keys[i]
    #         curr_student_idx = min(self.student_layer_head, len(student_keys) - 1)
    #         next_student_idx = min(self.student_layer_head + 1, len(student_keys) - 1)

    #         key_student_curr = student_keys[curr_student_idx]
    #         key_student_next = student_keys[next_student_idx]

    #         shape = self.teacher_weights[key_teacher].shape

    #         teacher_w = self._prepare_weights(self.teacher_weights[key_teacher], shape)
    #         student_curr_w = self._prepare_weights(self.student_weights[key_student_curr], shape)
    #         student_next_w = self._prepare_weights(self.student_weights[key_student_next], shape)

    #         result: str = get_closest_aligned_layer(student_curr_w, student_next_w, teacher_w)

    #         if result == "current" and mapping_count < self.max_mappings_per_layer:
    #             self.alignment_map[key_teacher] = key_student_curr
    #             mapping_count += 1
    #         else:
    #             self.alignment_map[key_teacher] = key_student_next
    #             mapping_count = 1
    #             self.student_layer_head += 1
    def _build_alignment_map(self) -> None:
        teacher_keys = list(self.teacher_weights.keys())
        student_keys = list(self.student_weights.keys())
        number_of_teacher_layers = len(teacher_keys)
        indices = torch.linspace(0, len(student_keys) - 1, steps=number_of_teacher_layers)
        # normal_samples = torch.randn(number_of_student_layers)
        # sorted_samples, _ = torch.sort(normal_samples)
        # normalized = (sorted_samples - sorted_samples.min()) / (sorted_samples.max() - sorted_samples.min() + 1e-8)
        # indices = (normalized * number_of_teacher_layers).long().clamp(0, number_of_teacher_layers - 1)
        # selected_teacher_keys = [teacher_keys[i] for i in indices.tolist()]
        selected_student_keys = [student_keys[int(i)] for i in indices]
        self.alignment_map = dict(zip(teacher_keys, selected_student_keys))


    def compute_layer_losses(self) -> Tensor: 
        layer_loss = torch.tensor(0.0, device=self.device)
        #for name, param in self.student.named_parameters():
            #print(f"{name}: grad shape = {param.grad.shape if param.grad is not None else None}")
        for teacher_key, student_key in self.alignment_map.items():
            # shape = self.teacher_weights[teacher_key].shape
            _,teacher_aligned,student_aligned = build_auto_matching_pair(
                self.teacher_weights[teacher_key].permute(3, 0, 1, 2),
                self.student_weights[student_key].permute(3, 0, 1, 2)
            )
            # teacher_w = self._prepare_weights(self.teacher_weights[teacher_key], shape)
            # student_w = self._prepare_weights(self.student_weights[student_key], shape)
            #print(f"teacher_w: {self.teacher_weights[teacher_key].shape}")
            # print(f"student_w: {student_aligned.shape}")
            # student_proj = self.proj_linear_layer(student_w)
            #teacher_proj = self.proj_linear_layer(teacher_w)
            #layer_loss += kd_loss(teacher_w, student_proj, temperature=2, reduce=True)
            # layer_loss += mean_pairwise_cosine_distance(student_proj, teacher_w)
            #T=[1,3,5,7,9,10]
            layer_loss+= kd_loss(student_aligned,teacher_aligned,self.Tem)
            #layer_loss += F.mse_loss(student_aligned, teacher_aligned)

        return self.align_layers * layer_loss
