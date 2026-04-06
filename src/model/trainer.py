from contextlib import nullcontext

import torch
from tqdm import tqdm


def _autocast_context(device, use_amp):
    enabled = bool(use_amp) and getattr(device, "type", "cpu") == "cuda"
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _compute_adherence_metrics(logits, labels, density_values, difficulty_values, ts_token_ids, pad_id=0):
    if ts_token_ids is None:
        return {}

    with torch.no_grad():
        pred_ids = torch.argmax(logits, dim=-1)
        valid_mask = labels != pad_id

        if valid_mask.sum().item() == 0:
            return {}

        ts_token_ids_set = set(int(x) for x in ts_token_ids)
        pred_event_density_proxy = []

        batch_size = pred_ids.size(0)
        for i in range(batch_size):
            valid_pred = pred_ids[i][valid_mask[i]]
            pred_token_count = max(1, int(valid_pred.numel()))
            pred_event_count = int(sum(1 for x in valid_pred.tolist() if int(x) not in ts_token_ids_set))
            pred_event_density_proxy.append(pred_event_count / float(pred_token_count))

        pred_density_proxy = torch.tensor(pred_event_density_proxy, dtype=torch.float32, device=labels.device)

        # Both are proxy metrics (logging-only):
        # - density_proxy_abs_error measures how close generated event intensity is to target density condition
        # - difficulty_proxy_drift tracks mismatch between difficulty control and generated event intensity
        density_proxy_abs_error = (pred_density_proxy - density_values.float()).abs().mean().item()
        difficulty_proxy_drift = (pred_density_proxy - difficulty_values.float()).abs().mean().item()

        return {
            "density_proxy_abs_error": density_proxy_abs_error,
            "difficulty_proxy_drift": difficulty_proxy_drift,
        }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    adherence_config=None,
    scaler=None,
    use_amp=False,
):
    model.train()

    total_loss = 0.0
    total_batches = 0
    total_density_error = 0.0
    total_difficulty_drift = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)

    ts_token_ids = None
    pad_id = 0
    if adherence_config is not None:
        ts_token_ids = adherence_config.get("ts_token_ids")
        pad_id = int(adherence_config.get("pad_id", 0))

    for batch in pbar:
        audio = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        difficulty_values = batch["difficulty_values"].to(device)
        density_values = batch["density_values"].to(device)
        beatmap_id_values = batch["beatmap_id_values"].to(device)

        optimizer.zero_grad()

        with _autocast_context(device, use_amp):
            logits = model(
                audio=audio,
                input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                difficulty_values=difficulty_values,
                density_values=density_values,
                beatmap_id_values=beatmap_id_values,
            )

            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if scaler is not None and bool(use_amp) and getattr(device, "type", "cpu") == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        metrics = _compute_adherence_metrics(
            logits=logits,
            labels=labels,
            density_values=density_values,
            difficulty_values=difficulty_values,
            ts_token_ids=ts_token_ids,
            pad_id=pad_id,
        )

        total_loss += loss.item()
        total_batches += 1
        if metrics:
            total_density_error += metrics["density_proxy_abs_error"]
            total_difficulty_drift += metrics["difficulty_proxy_drift"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dens_err=f"{metrics['density_proxy_abs_error']:.3f}",
                diff_drift=f"{metrics['difficulty_proxy_drift']:.3f}",
            )
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_batches
    out = {"loss": avg_loss}
    if ts_token_ids is not None and total_batches > 0:
        out["density_proxy_abs_error"] = total_density_error / total_batches
        out["difficulty_proxy_drift"] = total_difficulty_drift / total_batches
    return out


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    adherence_config=None,
    use_amp=False,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0
    total_density_error = 0.0
    total_difficulty_drift = 0.0
    pbar = tqdm(dataloader, desc="Validation", leave=False)

    ts_token_ids = None
    pad_id = 0
    if adherence_config is not None:
        ts_token_ids = adherence_config.get("ts_token_ids")
        pad_id = int(adherence_config.get("pad_id", 0))

    for batch in pbar:
        audio = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        difficulty_values = batch["difficulty_values"].to(device)
        density_values = batch["density_values"].to(device)
        beatmap_id_values = batch["beatmap_id_values"].to(device)

        with _autocast_context(device, use_amp):
            logits = model(
                audio=audio,
                input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                difficulty_values=difficulty_values,
                density_values=density_values,
                beatmap_id_values=beatmap_id_values,
            )

            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        metrics = _compute_adherence_metrics(
            logits=logits,
            labels=labels,
            density_values=density_values,
            difficulty_values=difficulty_values,
            ts_token_ids=ts_token_ids,
            pad_id=pad_id,
        )

        total_loss += loss.item()
        total_batches += 1
        if metrics:
            total_density_error += metrics["density_proxy_abs_error"]
            total_difficulty_drift += metrics["difficulty_proxy_drift"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dens_err=f"{metrics['density_proxy_abs_error']:.3f}",
                diff_drift=f"{metrics['difficulty_proxy_drift']:.3f}",
            )
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_batches
    out = {"loss": avg_loss}
    if ts_token_ids is not None and total_batches > 0:
        out["density_proxy_abs_error"] = total_density_error / total_batches
        out["difficulty_proxy_drift"] = total_difficulty_drift / total_batches
    return out


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=50,
    scheduler=None,
    adherence_config=None,
    scaler=None,
    use_amp=False,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "train_density_proxy_abs_error": [],
        "val_density_proxy_abs_error": [],
        "train_difficulty_proxy_drift": [],
        "val_difficulty_proxy_drift": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            adherence_config=adherence_config,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_stats = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            adherence_config=adherence_config,
            use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["lr"].append(current_lr)

        if "density_proxy_abs_error" in train_stats:
            history["train_density_proxy_abs_error"].append(train_stats["density_proxy_abs_error"])
            history["val_density_proxy_abs_error"].append(val_stats["density_proxy_abs_error"])
            history["train_difficulty_proxy_drift"].append(train_stats["difficulty_proxy_drift"])
            history["val_difficulty_proxy_drift"].append(val_stats["difficulty_proxy_drift"])

        print(
            f"Epoch {epoch}/{num_epochs} | lr: {current_lr:.6f} | "
            f"train loss: {train_stats['loss']:.4f} | val loss: {val_stats['loss']:.4f}"
        )
        if "density_proxy_abs_error" in train_stats:
            print(
                f"  adherence | train density err: {train_stats['density_proxy_abs_error']:.3f} | "
                f"val density err: {val_stats['density_proxy_abs_error']:.3f} | "
                f"train difficulty drift: {train_stats['difficulty_proxy_drift']:.3f} | "
                f"val difficulty drift: {val_stats['difficulty_proxy_drift']:.3f}"
            )

        if scheduler is not None:
            scheduler.step()

    return history


def plot_loss(history):
    import matplotlib.pyplot as plt

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2, color="#1f77b4")
    plt.axvline(
        x=8,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Best Epoch",
    )
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2, color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
