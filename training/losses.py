from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


def attack_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    pred = pred.view(-1).clamp(1e-6, 1 - 1e-6)
    target = target.float().view(-1)
    weights = torch.where(
        target > 0.5,
        torch.full_like(target, float(pos_weight)),
        torch.ones_like(target),
    )
    return F.binary_cross_entropy(pred, target, weight=weights)


def stage_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    target = target.long().view(-1)
    if mask is not None and float(mask.sum().item()) <= 0:
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits, target, reduction="none")
    if mask is not None:
        mask = mask.float().view(-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def target_loss(attention: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    if attention.dim() == 1:
        attention = attention.unsqueeze(0)
    return F.nll_loss(torch.log(attention + 1e-9), target_index.long().view(-1))


def state_prediction_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(predicted, observed.detach())


def rollout_loss(
    rollout_preds: List[Dict[str, torch.Tensor]],
    attack_targets: List[torch.Tensor],
    stage_targets: List[torch.Tensor],
    target_indices: Optional[List[Optional[torch.Tensor]]] = None,
    state_targets: Optional[List[torch.Tensor]] = None,
    stage_masks: Optional[List[torch.Tensor]] = None,
    state_weight: float = 0.25,
    attack_weight: float = 1.0,
    stage_weight: float = 1.0,
    target_weight: float = 1.0,
    rollout_decay: float = 0.8,
    attack_pos_weight: float = 1.0,
) -> torch.Tensor:
    total = None
    for i, pred in enumerate(rollout_preds):
        w = rollout_decay ** i
        current = (
            attack_weight * attack_loss(
                pred["attack_probability"],
                attack_targets[i],
                pos_weight=attack_pos_weight,
            )
            + stage_weight * stage_loss(
                pred["stage_logits"],
                stage_targets[i],
                mask=stage_masks[i] if stage_masks is not None else None,
            )
        )
        if state_targets is not None and pred.get("latent_state") is not None and i < len(state_targets):
            current = current + state_weight * state_prediction_loss(
                pred["latent_state"], state_targets[i]
            )
        if target_indices is not None and i < len(target_indices):
            target = target_indices[i]
            if target is not None and pred.get("target_attention") is not None:
                current = current + target_weight * target_loss(
                    pred["target_attention"], target
                )
        current = w * current
        total = current if total is None else total + current
    if total is None:
        return torch.tensor(0.0, requires_grad=True)
    return total
