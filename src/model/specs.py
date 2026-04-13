from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _required_context_max_len(
    history_max_tokens: int,
    retrieval_top_k: int,
    retrieval_max_tokens_per_window: int,
    use_motif_retrieval: bool,
    current_window_budget: int = 128,
) -> int:
    retrieval_budget = 0
    if use_motif_retrieval:
        retrieval_budget = max(0, int(retrieval_top_k)) * max(1, int(retrieval_max_tokens_per_window))
    return max(1, int(history_max_tokens)) + retrieval_budget + max(1, int(current_window_budget))


@dataclass
class ArchitectureSpec:
    name: str = "taiko_transformer"
    input_dim: int = 128
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    history_max_tokens: int = 1024
    retrieval_top_k: int = 2
    retrieval_max_tokens_per_window: int = 64
    retrieval_exclude_last_n_windows: int = 2
    use_motif_retrieval: bool = True
    max_cached_charts: int = 4

    def __post_init__(self) -> None:
        self.history_max_tokens = max(1, int(self.history_max_tokens))
        self.retrieval_top_k = max(0, int(self.retrieval_top_k))
        self.retrieval_max_tokens_per_window = max(1, int(self.retrieval_max_tokens_per_window))
        self.retrieval_exclude_last_n_windows = max(0, int(self.retrieval_exclude_last_n_windows))
        self.max_cached_charts = max(1, int(self.max_cached_charts))
        self.max_len = max(1, int(self.max_len))

        if self.name == "taiko_context_transformer":
            self.max_len = max(self.max_len, self.required_context_max_len())

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_len": self.max_len,
        }

    def context_kwargs(self) -> dict[str, Any]:
        return {
            "history_max_tokens": self.history_max_tokens,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_max_tokens_per_window": self.retrieval_max_tokens_per_window,
            "retrieval_exclude_last_n_windows": self.retrieval_exclude_last_n_windows,
            "use_motif_retrieval": self.use_motif_retrieval,
        }

    def dataset_context_kwargs(self) -> dict[str, Any]:
        return {
            "max_cached_charts": self.max_cached_charts,
        }

    def required_context_max_len(self) -> int:
        return _required_context_max_len(
            history_max_tokens=self.history_max_tokens,
            retrieval_top_k=self.retrieval_top_k,
            retrieval_max_tokens_per_window=self.retrieval_max_tokens_per_window,
            use_motif_retrieval=self.use_motif_retrieval,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchitectureSpec":
        return cls(**data)


@dataclass
class TrainingSpec:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingSpec":
        return cls(**data)
