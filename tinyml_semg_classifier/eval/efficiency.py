from __future__ import annotations
import math
import time

import torch
from torch import nn


def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 50,
    iters: int = 200,
) -> float:
    device = input_tensor.device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    if was_training:
        model.train()
    return (elapsed / max(1, iters)) * 1000.0


def measure_efficiency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    latency_warmup: int = 50,
    latency_iters: int = 200,
    include_latency: bool = True,
) -> dict:
    params = count_parameters(model)
    macs, _ = estimate_macs_flops(model, input_tensor)
    latency_ms = None
    if include_latency:
        latency_ms = measure_latency(
            model,
            input_tensor,
            warmup=latency_warmup,
            iters=latency_iters,
        )
    return {"params": params, "macs": macs, "latency_ms": latency_ms}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_macs_flops(model, input_tensor):
    """Estimate MACs/FLOPs for one forward pass (conv/linear only)."""
    macs = 0
    handles = []

    def conv_hook(module, _inputs, output):
        nonlocal macs
        out = output[0] if isinstance(output, (tuple, list)) else output
        if out is None:
            return
        kernel_ops = math.prod(module.kernel_size) * (
            module.in_channels // module.groups
        )
        macs += out.numel() * kernel_ops

    def linear_hook(module, _inputs, output):
        nonlocal macs
        out = output[0] if isinstance(output, (tuple, list)) else output
        if out is None:
            return
        macs += out.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    flops = macs * 2
    return macs, flops
