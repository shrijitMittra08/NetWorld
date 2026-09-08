from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

from .gradient_attribution import integrated_gradients_attribution

def shap_feature_attribution(
    model: Any,
    background: pd.DataFrame,
    sample: pd.DataFrame,
) -> Dict[str, float]:
    """
    SHAP-based feature attribution for tabular inputs.
    Returns a feature->importance mapping.
    """
    if shap is None:
        return {}

    explainer = shap.Explainer(model, background)
    values = explainer(sample)

    if hasattr(values, "values"):
        raw = values.values
        if raw.ndim == 2:
            importance = np.abs(raw).mean(axis=0)
        else:
            importance = np.abs(raw)
        return {
            str(col): float(score)
            for col, score in zip(sample.columns, importance)
        }

    return {}

def gradient_feature_attribution(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    target: Optional[int] = None,
) -> Dict[str, float]:
    """
    Gradient-based attribution over tensor inputs.
    """
    attr = integrated_gradients_attribution(model, input_tensor, target=target)
    scores = attr.abs().mean(dim=0).view(-1)

    if feature_names and len(feature_names) == len(scores):
        return {name: float(score) for name, score in zip(feature_names, scores.tolist())}

    return {f"feature_{i}": float(v) for i, v in enumerate(scores.tolist())}