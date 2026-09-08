from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

def attack_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.float().view(-1)
    return F.binary_cross_entropy(pred, target)

def stage_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return F.cross_entropy(logits, target.long().view(-1))

def target_loss(attention: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    if attention.dim() == 1:
        attention = attention.unsqueeze(0)
    return F.nll_loss(torch.log(attention + 1e-9), target_index.long().view(-1))

def rollout_loss(
    rollout_preds: List[Dict[str, torch.Tensor]],
    attack_targets: List[torch.Tensor],
    stage_targets: List[torch.Tensor],
    target_indices: Optional[List[torch.Tensor]] = None,
    attack_weight: float = 1.0,
    stage_weight: float = 1.0,
    target_weight: float = 1.0,
    rollout_decay: float = 0.8,
) -> torch.Tensor:
    total = torch.tensor(0.0, requires_grad=True)
    for i, pred in enumerate(rollout_preds):
        w = rollout_decay ** i
        total = total + w * (
            attack_weight * attack_loss(pred["attack_probability"], attack_targets[i])
            + stage_weight * stage_loss(pred["stage_logits"], stage_targets[i])
        )
        if target_indices is not None and pred.get("target_attention") is not None:
            total = total + w * target_weight * target_loss(
                pred["target_attention"], target_indices[i]
            )
    return total