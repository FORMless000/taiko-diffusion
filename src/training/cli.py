from __future__ import annotations

import argparse
import json

from src.model.specs import ModelSpec

from .config import OptimizationConfig, TrainingRunConfig
from .pipeline import train_from_raw_osz


def main() -> None:
    parser = argparse.ArgumentParser(description="Train taiko-diffusion models from raw .osz files.")
    parser.add_argument("raw_osz", nargs="*", help="Paths to raw .osz files")
    parser.add_argument("--run-dir", required=True, help="Output run directory")
    parser.add_argument("--resume-checkpoint", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--architecture", default="transformer_baseline", help="Registered architecture name")
    parser.add_argument("--model-spec-json", default=None, help="JSON object for model parameters")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs for the run")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", default=None, help="Explicit torch device, e.g. cpu or cuda")
    parser.add_argument("--use-amp", action="store_true", help="Enable CUDA AMP training")
    args = parser.parse_args()

    model_params = json.loads(args.model_spec_json) if args.model_spec_json else {}

    config = TrainingRunConfig(
        run_dir=args.run_dir,
        raw_osz_paths=list(args.raw_osz),
        model_spec=ModelSpec(name=args.architecture, params=model_params),
        optimization=OptimizationConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            use_amp=args.use_amp,
        ),
        resume_checkpoint=args.resume_checkpoint,
        device=args.device,
    )

    result = train_from_raw_osz(config)
    print(json.dumps(result, indent=2))
