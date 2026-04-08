from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Callable


@dataclass
class WandbConfig:
    enabled: bool = False
    run_name: str = "default"
    log_every_n_batches: int = 100
    notebook_name: str = ""
    offline: bool = False
    api_key: str = ""
    project: str = "taiko-transformer"
    entity: str = "yiy523-lehigh-university"
    mode_name_for_run: str = ""
    wandb_dir: str = ""


@dataclass
class WandbRuntime:
    run: Any | None
    metrics_logger: Callable[[dict[str, Any]], None] | None


def setup_wandb_runtime(config: WandbConfig, *, model_name: str) -> WandbRuntime:
    if not config.enabled:
        return WandbRuntime(run=None, metrics_logger=None)

    # Lazy import: wandb is never imported unless explicitly enabled.
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging requested, but `wandb` is not installed. Install optional deps with `pip install -e .[wandb]`."
        ) from exc

    notebook_name = config.notebook_name.strip()
    if notebook_name:
        os.environ["WANDB_NOTEBOOK_NAME"] = notebook_name
    os.environ.setdefault("WANDB_NOTEBOOK_NAME", "train.py")

    if config.wandb_dir.strip():
        os.environ["WANDB_DIR"] = config.wandb_dir.strip()
    else:
        default_wandb_dir = Path.cwd() / ".wandb"
        default_wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(default_wandb_dir.resolve()))

    if config.offline:
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ.pop("WANDB_MODE", None)
        api_key = config.api_key.strip() or os.environ.get("WANDB_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "W&B online logging requested, but no API key was provided. "
                "Pass api_key in WandbConfig, set WANDB_API_KEY, or enable offline mode."
            )
        os.environ["WANDB_API_KEY"] = api_key
        if hasattr(wandb, "login"):
            wandb.login(key=api_key, relogin=False)

    mode_name = config.mode_name_for_run.strip() or model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = wandb.init(
        project=config.project,
        name=f" taiko-transformer_run_{mode_name}_{config.run_name}_{timestamp}",
        entity=config.entity,
    )
    if hasattr(run, "define_metric"):
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
    else:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    return WandbRuntime(run=run, metrics_logger=run.log)
