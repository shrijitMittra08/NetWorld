from __future__ import annotations

from typing import Any, Dict, Optional
import torch
try:
    from captum.attr import IntegratedGradients
except Exception:  # pragma: no cover
    IntegratedGradients = None

def integrated_gradients_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: Optional[int] = None,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns attribution scores for tensor inputs.
    If Captum is unavailable, falls back to gradient magnitude.
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    if IntegratedGradients is None:
        inputs = inputs.clone().detach().requires_grad_(True)
        output = model(inputs)
        if target is not None and output.dim() > 1:
            output = output[..., target]
        output.sum().backward()
        return inputs.grad.abs().detach()

    ig = IntegratedGradients(model)
    attr = ig.attribute(inputs, baselines=baseline, target=target)
    return attr.detach()