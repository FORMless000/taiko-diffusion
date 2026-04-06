from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.model.specs import ModelSpec


def _load_dataclass(data: Mapping[str, Any] | None, cls):
    if data is None:
        return cls()
    kwargs = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
    return cls(**kwargs)


@dataclass
class PreprocessingConfig:
    include_bpm_events: bool = False
    reject_offgrid_notes: bool = True
    offgrid_tolerance_ms: float = 5.0
    keep_only_max_notes_per_song: bool = False
    overwrite_unpacked: bool = False
    overwrite_parsed: bool = False


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_state: int = 42


@dataclass
class OptimizationConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = False
    checkpoint_every_epochs: int = 1
    scheduler_name: str = "none"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 1.0
    seed: int = 42


@dataclass
class TrainingRunConfig:
    run_dir: str
    raw_osz_paths: list[str] = field(default_factory=list)
    model_spec: ModelSpec = field(default_factory=ModelSpec)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    resume_checkpoint: str | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "raw_osz_paths": list(self.raw_osz_paths),
            "model_spec": self.model_spec.to_dict(),
            "preprocessing": asdict(self.preprocessing),
            "split": asdict(self.split),
            "optimization": asdict(self.optimization),
            "resume_checkpoint": self.resume_checkpoint,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingRunConfig":
        return cls(
            run_dir=str(data["run_dir"]),
            raw_osz_paths=[str(x) for x in data.get("raw_osz_paths", [])],
            model_spec=ModelSpec.from_dict(data.get("model_spec")),
            preprocessing=_load_dataclass(data.get("preprocessing"), PreprocessingConfig),
            split=_load_dataclass(data.get("split"), SplitConfig),
            optimization=_load_dataclass(data.get("optimization"), OptimizationConfig),
            resume_checkpoint=data.get("resume_checkpoint"),
            device=data.get("device"),
        )
