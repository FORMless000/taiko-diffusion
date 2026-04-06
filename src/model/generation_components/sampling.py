from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class SamplingConfig:
    temperature: float = 0.9
    top_p: float = 0.82
    top_k: int = 4
    ts_top_k: int = 2
    min_event_candidates: int = 2
    repetition_penalty: float = 1.0


def apply_repetition_penalty(logits, generated_ids, repetition_penalty: float):
    if repetition_penalty is None or repetition_penalty <= 1.0 or generated_ids.numel() == 0:
        return logits

    adjusted = logits.clone()
    unique_ids = torch.unique(generated_ids)
    for token_id in unique_ids:
        idx = int(token_id.item())
        if adjusted[idx] < 0:
            adjusted[idx] *= repetition_penalty
        else:
            adjusted[idx] /= repetition_penalty
    return adjusted


def class_aware_candidate_ids(
    logits_1d,
    *,
    event_token_ids_tensor,
    ts_token_ids_tensor,
    eos_id: int,
    top_k: int,
    ts_top_k: int,
    min_event_candidates: int,
):
    if event_token_ids_tensor.numel() == 0:
        return torch.arange(logits_1d.size(0), device=logits_1d.device)

    event_k = max(int(min_event_candidates), min(int(top_k), int(event_token_ids_tensor.numel())))
    if event_k <= 0:
        event_k = min_event_candidates

    event_scores = logits_1d.index_select(0, event_token_ids_tensor)
    _, event_local_idx = torch.topk(event_scores, k=event_k)
    event_ids = event_token_ids_tensor.index_select(0, event_local_idx)

    if ts_token_ids_tensor.numel() > 0 and ts_top_k > 0:
        ts_k = min(int(ts_top_k), int(ts_token_ids_tensor.numel()))
        ts_scores = logits_1d.index_select(0, ts_token_ids_tensor)
        _, ts_local_idx = torch.topk(ts_scores, k=ts_k)
        ts_ids = ts_token_ids_tensor.index_select(0, ts_local_idx)
        candidates = torch.cat([event_ids, ts_ids], dim=0)
    else:
        candidates = event_ids

    if eos_id not in candidates.tolist():
        candidates = torch.cat(
            [candidates, torch.tensor([eos_id], dtype=torch.long, device=logits_1d.device)],
            dim=0,
        )

    return torch.unique(candidates)


def apply_top_p(probs_1d, token_ids, top_p: float):
    top_p = float(max(0.05, min(1.0, top_p)))

    sorted_probs, sorted_idx = torch.sort(probs_1d, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    keep_sorted = cumulative <= top_p
    keep_sorted[0] = True

    kept_idx = sorted_idx[keep_sorted]
    kept_ids = token_ids.index_select(0, kept_idx)
    kept_probs = probs_1d.index_select(0, kept_idx)
    kept_probs = kept_probs / kept_probs.sum()
    return kept_ids, kept_probs


def sample_next_token(
    logits_last,
    generated_ids,
    sampling_config: SamplingConfig,
    *,
    event_token_ids_tensor,
    ts_token_ids_tensor,
    eos_id: int,
):
    temperature = float(sampling_config.temperature)
    top_p = float(sampling_config.top_p)
    top_k = int(max(1, sampling_config.top_k))
    ts_top_k = int(max(0, sampling_config.ts_top_k))
    min_event_candidates = int(max(1, sampling_config.min_event_candidates))

    logits_1d = logits_last[0]
    logits_1d = apply_repetition_penalty(
        logits_1d,
        generated_ids,
        repetition_penalty=float(sampling_config.repetition_penalty),
    )

    if temperature <= 0.0:
        return torch.argmax(logits_1d).view(1, 1)

    scaled_logits = logits_1d / max(1e-6, temperature)
    candidate_ids = class_aware_candidate_ids(
        scaled_logits,
        event_token_ids_tensor=event_token_ids_tensor,
        ts_token_ids_tensor=ts_token_ids_tensor,
        eos_id=eos_id,
        top_k=top_k,
        ts_top_k=ts_top_k,
        min_event_candidates=min_event_candidates,
    )

    candidate_logits = scaled_logits.index_select(0, candidate_ids)
    candidate_probs = torch.softmax(candidate_logits, dim=0)

    filtered_ids, filtered_probs = apply_top_p(candidate_probs, candidate_ids, top_p)

    event_mask = torch.isin(filtered_ids, event_token_ids_tensor)
    if event_mask.sum().item() < min_event_candidates:
        event_candidates = event_token_ids_tensor
        event_scores = scaled_logits.index_select(0, event_candidates)
        _, extra_idx = torch.topk(
            event_scores,
            k=min(min_event_candidates, int(event_candidates.numel())),
        )
        extra_ids = event_candidates.index_select(0, extra_idx)

        merged_ids = torch.unique(torch.cat([filtered_ids, extra_ids], dim=0))
        merged_logits = scaled_logits.index_select(0, merged_ids)
        merged_probs = torch.softmax(merged_logits, dim=0)
        filtered_ids, filtered_probs = apply_top_p(merged_probs, merged_ids, top_p=1.0)

    sampled_local = torch.multinomial(filtered_probs, num_samples=1)
    return filtered_ids.index_select(0, sampled_local).view(1, 1)
