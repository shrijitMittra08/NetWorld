from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from .schema import canonicalFieldNames

def writeParquet(
        df: pd.DataFrame,
        outputPath: str | Path,
        filename: Optional[str] = None,
        index: bool = False,
) -> Path:
    outputDir = Path(outputPath)
    outputDir.mkdir(parents=True, exist_ok=True)

    outName = filename or "dataset.parquet"
    parquetPath = outputDir/outName

    missing = [col for col in canonicalFieldNames() if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot write parquet: missing canonical columns: {missing}")

    canonicalDf = df[canonicalFieldNames()].copy()
    extraCols = [c for c in df.columns if c not in canonicalFieldNames()]
    if extraCols:
        canonicalDf = pd.concat([canonicalDf, df[extraCols]], axis=1)

    canonicalDf.to_parquet(parquetPath, index=index)
    return parquetPath

def writeParquetPartitioned(
    df: pd.DataFrame,
    outputDir: str | Path,
    partitionCol: str,
    index: bool = False,   
) -> list[Path]:
    if partitionCol not in df.columns:
        raise ValueError(f"Partition column not found: {partitionCol}")

    baseDir = Path(outputDir)
    baseDir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for value, group in df.groupby(partitionCol, dropna=False):
        safeValue = "null" if pd.isna(value) else str(value).replace("/", "_")
        filePath = baseDir/f"{partitionCol}={safeValue}.parquet"
        group.to_parquet(filePath, index=index)
        written.append(filePath)

    return written