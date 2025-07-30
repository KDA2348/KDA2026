import torch
import torch.nn as nn

def compute_kernel_stride(in_size, out_size):
    assert out_size <= in_size, f"out_size ({out_size}) must be <= in_size ({in_size})"

    stride = max(in_size // out_size, 1)
    kernel_size = in_size - (out_size - 1) * stride

    if kernel_size <= 0:
        stride = 1
        kernel_size = in_size - (out_size - 1) * stride

    assert kernel_size > 0, f"Invalid kernel size computed: {kernel_size}"
    return kernel_size, stride

def match_one_tensor_to_another(source_tensor: torch.Tensor, target_tensor: torch.Tensor):
    n1, c1, h1, w1 = source_tensor.shape
    n2, c2, h2, w2 = target_tensor.shape

    kernel_h, stride_h = compute_kernel_stride(h1, h2)
    kernel_w, stride_w = compute_kernel_stride(w1, w2)

    conv = nn.Conv2d(
        in_channels=c1,
        out_channels=c2,
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=0,
        bias=False
    ).to(source_tensor.device)

    output = conv(source_tensor)
    assert output.shape == target_tensor.shape, f"Output shape {output.shape} does not match target {target_tensor.shape}"
    return conv, output

def build_auto_matching_pair(x1: torch.Tensor, x2: torch.Tensor):
    
    assert x1.ndim == 4 and x2.ndim == 4, "Inputs must be 4D tensors"

    n1, c1, h1, w1 = x1.shape
    n2, c2, h2, w2 = x2.shape
    # print(x1.shape, x2.shape)
    # print("********")

    if n1 < n2:
        if n1 == 1:
            x1 = x1.expand(n2, -1, -1, -1)
        else:
            reps = n2 // n1 + int(n2 % n1 != 0)
            x1 = x1.repeat(reps, 1, 1, 1)[:n2]
        n1 = n2
    elif n2 <n1:
        if n2 == 1:
            x2 = x2.expand(n1, -1, -1, -1)
        else:
            reps = n1 // n2 + int(n1 % n2 != 0)
            x2 = x2.repeat(reps, 1, 1, 1)[:n1]
        n2 = n1
    
    if w1 < w2:
        if w1 == 1:
            x1 = x1.expand( -1, -1, -1, w2)
        else:
            reps = w2 // w1 + int(w2 % w1 != 0)
            x1 = x1.repeat( 1, 1, 1, reps)[:w2]
        w1 = w2
    elif w2 <w1:
        if w2 == 1:
            x2 = x2.expand( -1, -1, -1, w1)
        else:
            reps = w1 // w2 + int(w1 % w2 != 0)
            x2 = x2.repeat( 1, 1, 1, reps)[:w1]
        w2 = w1

    # if w1 == 1 and w2 > 1:
    #     x1 = x1.expand(-1, -1, -1, w2)
    #     w1 = w2
    # elif w2 == 1 and w1 > 1:
    #     x2 = x2.expand(-1, -1, -1, w1)
    #     w2 = w1
    assert n1 == n2, f"Batch size mismatch: {n1} vs {n2}"

    s1 = x1.shape
    s2 = x2.shape
    # print("sss")
    # print(s1, s2)
    if s1[2] * s1[3] > s2[2] * s2[3]:
        conv, x1_aligned = match_one_tensor_to_another(x1, x2)
        return conv, x1_aligned, x2
    else:
        conv, x2_aligned = match_one_tensor_to_another(x2, x1)
        return conv, x1, x2_aligned
    