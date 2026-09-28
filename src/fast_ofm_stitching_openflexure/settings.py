"""Validated resource and registration settings for the worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StitchingProcessSettings:
    """Bound native concurrency, RAM cache use, and correlation implementation."""

    workers: int = 3
    ram_cache_mb: int = 4096
    correlation_mode: Literal["full", "overlap-strip"] = "full"

    def __post_init__(self) -> None:
        if not 1 <= self.workers <= 64:
            raise ValueError("workers must be in [1, 64]")
        if self.ram_cache_mb < 64:
            raise ValueError("ram_cache_mb must be at least 64")
        if self.correlation_mode not in ("full", "overlap-strip"):
            raise ValueError("unsupported correlation mode")
