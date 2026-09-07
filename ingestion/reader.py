from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Union

import pandas as pd

class Reader:

    def __init__(
            self,
            inputPath: Union[str, Path],
            encoding: Optional[str] = None,
            lowMemory: bool = False,
    ) -> None:
        self.inputPath = Path(inputPath)
        self.encoding = encoding
        self.lowMemory = lowMemory

    def csvFiles(self) -> list[Path]:
        if not self.inputPath.exists():
            raise FileNotFoundError(f"Input path doesn't exist: {self.inputPath}")
        
        if (self.inputPath.is_file()):
            if self.inputPath.suffix.lower() != ".csv":
                raise ValueError(f"Expected csv file, got: {self.inputPath}")
            return [self.inputPath]

        csv_files = sorted(self.inputPath.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No csv files found: {self.inputPath}")
        return csv_files

    def readCsv(self, filePath: Union[str, Path]) -> pd.DataFrame:
        filePath = Path(filePath)
        if not filePath.exists():
            raise FileNotFoundError(f"CSV file not found: {filePath}")

        read_kwargs = {
            "low_memory": self.lowMemory,
        }
        if self.encoding is not None:
            read_kwargs["encoding"] = self.encoding

        try:
            df = pd.read_csv(filePath, **read_kwargs)

        except UnicodeDecodeError:
            if self.encoding is not None:
                raise
            df = pd.read_csv(filePath, encoding="latin1", low_memory=self.lowMemory)

        df["network.source_file"] = filePath.name
        return df

    def iterDataframes(self) -> Iterator[pd.DataFrame]:
        for csv_file in self.csvFiles():
            yield self.readCsv(csv_file)

    def load(self) -> pd.DataFrame:
        frames = list(self.iterDataframes())
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)
