from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_ALLOWED_PRECISIONS = {"auto", "fp32", "bf16", "fp16"}


@dataclass(frozen=True)
class PrecisionRuntime:
    requested: str
    resolved: str
    autocast_enabled: bool
    autocast_dtype: torch.dtype | None
    scaler_enabled: bool
    fallback_reason: str = ""


def normalize_precision(precision: str | None) -> str:
    value = str(precision or "auto").strip().lower()
    if value not in _ALLOWED_PRECISIONS:
        raise ValueError(f"Unsupported precision '{precision}'. Allowed: {sorted(_ALLOWED_PRECISIONS)}")
    return value


def resolve_precision_runtime(precision: str | None, device: torch.device) -> PrecisionRuntime:
    requested = normalize_precision(precision)

    if device.type != "cuda":
        fallback_reason = ""
        if requested != "fp32":
            fallback_reason = f"{requested} is only enabled on CUDA; falling back to fp32 on {device.type}."
        return PrecisionRuntime(
            requested=requested,
            resolved="fp32",
            autocast_enabled=False,
            autocast_dtype=None,
            scaler_enabled=False,
            fallback_reason=fallback_reason,
        )

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    fallback_reason = ""

    if requested == "auto":
        resolved = "bf16" if bf16_supported else "fp16"
    elif requested == "bf16":
        if bf16_supported:
            resolved = "bf16"
        else:
            resolved = "fp16"
            fallback_reason = "CUDA bf16 is not supported on this device; falling back to fp16."
    else:
        resolved = requested

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    autocast_dtype = dtype_map.get(resolved)
    return PrecisionRuntime(
        requested=requested,
        resolved=resolved,
        autocast_enabled=autocast_dtype is not None,
        autocast_dtype=autocast_dtype,
        scaler_enabled=(resolved == "fp16"),
        fallback_reason=fallback_reason,
    )


def build_grad_scaler(precision_runtime: PrecisionRuntime):
    if not precision_runtime.scaler_enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def build_dataloader_runtime_kwargs(training_spec, device: torch.device) -> dict[str, Any]:
    num_workers = max(0, int(getattr(training_spec, "num_workers", 0)))

    pin_memory_raw = getattr(training_spec, "pin_memory", None)
    pin_memory = (device.type == "cuda") if pin_memory_raw is None else bool(pin_memory_raw)

    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers <= 0:
        return kwargs

    persistent_workers_raw = getattr(training_spec, "persistent_workers", None)
    persistent_workers = True if persistent_workers_raw is None else bool(persistent_workers_raw)
    prefetch_factor_raw = getattr(training_spec, "prefetch_factor", None)
    prefetch_factor = 2 if prefetch_factor_raw is None else max(1, int(prefetch_factor_raw))

    kwargs["persistent_workers"] = persistent_workers
    kwargs["prefetch_factor"] = prefetch_factor
    return kwargs
