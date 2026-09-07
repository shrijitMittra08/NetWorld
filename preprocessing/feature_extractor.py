from __future__ import annotations

from typing import Optional
import pandas as pd

from .normalizer import normalizeDataset, validateCanonicalDataFrame

def extractFeatures(
    df: pd.DataFrame,
    source_file: Optional[str] = None,
) -> pd.DataFrame:
    canonical_df = normalizeDataset(df, sourceFile=source_file)
    validateCanonicalDataFrame(canonical_df)
    return canonical_df