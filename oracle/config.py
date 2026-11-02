from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OracleConfig:
    """Training, validation, and signal policy configuration."""

    symbol: str = "BTC-USD"
    lookback_days: int = 729
    horizons: dict[str, int] = field(default_factory=lambda: {"4 hours": 4, "24 hours": 24, "3 days": 72})
    sequence_length: int = 48
    validation_fraction: float = 0.20
    interval_z: float = 1.28
    minimum_confidence: float = 0.40
    minimum_signal_sigma: float = 0.30
    refresh_seconds: int = 300
    random_state: int = 42
    history_hours: int = 168
    minimum_training_rows: int = 500

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if not self.horizons or any(hours < 1 for hours in self.horizons.values()):
            raise ValueError("at least one positive forecast horizon is required")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
