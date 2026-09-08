from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Union

import pandas as pd

from .authlog_reader import AuthLogReader
from .dataset_adapter import get_dataset_field_map
from .netflow_reader import NetFlowReader
from .pcap_reader import PcapReader
from .source_router import detect_source_type

class Reader:
    def __init__(
        self,
        inputPath: Union[str, Path],
        encoding: Optional[str] = None,
        lowMemory: bool = False,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.inputPath = Path(inputPath)
        self.encoding = encoding
        self.lowMemory = lowMemory
        self.dataset_name = dataset_name

    def csvFiles(self) -> list[Path]:
        if not self.inputPath.exists():
            raise FileNotFoundError(f"Input path doesn't exist: {self.inputPath}")

        if self.inputPath.is_file():
            if self.inputPath.suffix.lower() != ".csv":
                return [self.inputPath]
            return [self.inputPath]

        files = sorted(
            [p for p in self.inputPath.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".pcap", ".pcapng", ".log", ".txt", ".json", ".jsonl"}]
        )
        if not files:
            raise FileNotFoundError(f"No supported files found: {self.inputPath}")
        return files

    def readCsv(self, filePath: Union[str, Path]) -> pd.DataFrame:
        filePath = Path(filePath)
        if not filePath.exists():
            raise FileNotFoundError(f"CSV file not found: {filePath}")

        read_kwargs = {"low_memory": self.lowMemory}
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

    def read_any(self, filePath: Union[str, Path]) -> pd.DataFrame:
        filePath = Path(filePath)
        source_type = detect_source_type(filePath)

        if source_type == "csv":
            return self.readCsv(filePath)
        if source_type == "pcap":
            return PcapReader(filePath).parse()
        if source_type == "authlog":
            return AuthLogReader(filePath).parse()
        if source_type == "netflow":
            return NetFlowReader(filePath).parse()

        raise ValueError(f"Unsupported file type: {filePath}")

    def iterDataframes(self) -> Iterator[pd.DataFrame]:
        for path in self.csvFiles():
            yield self.read_any(path)

    def load(self) -> pd.DataFrame:
        frames = list(self.iterDataframes())
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)