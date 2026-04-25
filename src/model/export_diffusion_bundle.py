from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoints import export_diffusion_inference_bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a notebook-era diffusion checkpoint into an inference bundle.")
    parser.add_argument("--raw-checkpoint", required=True, help="Path to the raw training checkpoint (last.ckpt).")
    parser.add_argument("--vocab", required=True, help="Path to the collaborator-provided diffusion vocab (.pth).")
    parser.add_argument("--out", required=True, help="Output path for the inference bundle.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bundle_path = export_diffusion_inference_bundle(
        Path(args.out),
        raw_checkpoint_path=Path(args.raw_checkpoint),
        vocab_path=Path(args.vocab),
    )
    print(bundle_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
