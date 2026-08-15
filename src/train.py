"""Training loop shared by all four models."""

import os
import time

import torch
import torch.nn as nn

from src import config
from src.dataset import make_loader
from src.models import build_model
from src.utils import count_parameters, format_time, save_json


def _run_epoch(model, loader, optimizer, criterion, device, train: bool,
               teacher_forcing_ratio: float):
    model.train() if train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for src, src_len, tgt, _ in loader:
            src, tgt = src.to(device), tgt.to(device)
            if train:
                optimizer.zero_grad()

            tf_ratio = teacher_forcing_ratio if train else 0.0
            outputs = model(src, src_len, tgt, teacher_forcing_ratio=tf_ratio)

            # outputs: (batch, tgt_len, vocab) ; ignore t=0 (<sos>) on both sides
            output_dim = outputs.shape[-1]
            loss = criterion(
                outputs[:, 1:].reshape(-1, output_dim),
                tgt[:, 1:].reshape(-1),
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model_name, train_pairs, val_pairs, src_vocab, tgt_vocab, device,
                 num_epochs=config.NUM_EPOCHS, verbose=True):
    train_loader = make_loader(train_pairs, src_vocab, tgt_vocab, shuffle=True)
    val_loader = make_loader(val_pairs, src_vocab, tgt_vocab, shuffle=False)

    model = build_model(model_name, len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    n_params = count_parameters(model)
    if verbose:
        print(f"\n=== Training {model.display_name} ({n_params:,} trainable params) ===")

    history = {"train_loss": [], "val_loss": [], "epoch_seconds": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}.pt")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, device,
                                 train=True, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO)
        val_loss = _run_epoch(model, val_loader, optimizer, criterion, device,
                               train=False, teacher_forcing_ratio=0.0)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_seconds"].append(elapsed)

        if verbose:
            print(f"  epoch {epoch:2d}/{num_epochs} | train_loss {train_loss:.4f} "
                  f"| val_loss {val_loss:.4f} | {format_time(elapsed)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.PATIENCE:
                if verbose:
                    print(f"  early stopping (no val improvement for {config.PATIENCE} epochs)")
                break

    # reload best checkpoint before returning
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    history["num_params"] = n_params
    history["best_val_loss"] = best_val_loss
    save_json(history, os.path.join(config.METRICS_DIR, f"{model_name}_history.json"))
    return model, history
