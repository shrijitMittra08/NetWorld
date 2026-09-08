from __future__ import annotations

from typing import Optional
import pandas as pd

from .normalizer import normalizeDataset, validateCanonicalDataFrame

def extractFeatures(
    df: pd.DataFrame,
    source_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    canonical_df = normalizeDataset(
        df,
        sourceFile=source_file,
        dataset_name=dataset_name,
    )
    validateCanonicalDataFrame(canonical_df)
    return canonical_df