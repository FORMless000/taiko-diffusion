import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_batches = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        audio = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        optimizer.zero_grad()

        logits = model(
            audio=audio,
            input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_batches
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0
    pbar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in pbar:
        audio = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        logits = model(
            audio=audio,
            input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()
        total_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_batches
    return avg_loss


def fit(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=50, scheduler=None):
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch}/{num_epochs} | lr: {current_lr:.6f} | "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

    return history


def plot_loss(history):
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
