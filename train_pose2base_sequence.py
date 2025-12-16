#!/usr/bin/env python3
"""
Train a sequence-to-sequence transformer that maps a sequence of end-effector poses
(end_pos + Euler angles) to the corresponding sequence of base positions.

Key differences vs train_pose2base.py:
- Operates on grouped sequences: rows with the same group_id form one sequence.
- Sequences are ordered by point_index (ascending) and assumed to have equal length.
- Model outputs a full sequence of base_pos predictions (not a single timestep).

Usage example (project root):
  python -u train_pose2base_sequence.py \
    --model-name seq_exp \
    --epochs 200 \
    --batch-size 256 \
    --num-workers 8 \
    --amp auto \
    --save-every 10
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.amp as amp
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEFAULT_DATA_PATH = Path("./data/robot_training_data_seq_expanded.csv")
DEFAULT_MODELS_DIR = Path("./models")
DEFAULT_OUTPUTS_DIR = Path("./outputs")

INPUT_COLS = [
    "end_pos_x",
    "end_pos_y",
    "end_pos_z",
    "end_pose_x",
    "end_pose_y",
    "end_pose_z",
]
TARGET_COLS = ["base_pos_x", "base_pos_y", "base_pos_z"]
GROUP_COLS = ["group_id", "point_index"]


# =============== Utils ===============

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_features_from_df(df: pd.DataFrame) -> np.ndarray:
    """Convert a dataframe slice into features: [pos, sin(angles), cos(angles)]."""
    pos = df[["end_pos_x", "end_pos_y", "end_pos_z"]].to_numpy(dtype="float32")
    ang_deg = df[["end_pose_x", "end_pose_y", "end_pose_z"]].to_numpy(dtype="float32")
    ang_rad = np.deg2rad(ang_deg)
    ang_sin = np.sin(ang_rad)
    ang_cos = np.cos(ang_rad)
    feats = np.concatenate([pos, ang_sin, ang_cos], axis=1)  # (L, 9)
    return feats


def load_sequences(csv_path: Path, nrows: int | None = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load grouped sequences from CSV. Rows with the same group_id form a sequence.
    Sequences are ordered by point_index ascending. Groups with length mismatch
    are skipped to keep a fixed seq_len.

    Returns:
        X: (N, L, 9) float32
        y: (N, L, 3) float32
        seq_len: int
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    required = set(INPUT_COLS + TARGET_COLS + GROUP_COLS)
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=INPUT_COLS + TARGET_COLS + GROUP_COLS)
    df = df.sort_values(["group_id", "point_index"]).reset_index(drop=True)

    X_list, y_list = [], []
    seq_len = None
    for gid, g in df.groupby("group_id"):
        g = g.reset_index(drop=True)
        if seq_len is None:
            seq_len = len(g)
        if len(g) != seq_len:
            print(f"[warn] Skip group {gid}: length {len(g)} != expected {seq_len}")
            continue
        feats = build_features_from_df(g)
        targets = g[TARGET_COLS].to_numpy(dtype="float32")
        X_list.append(feats)
        y_list.append(targets)

    if not X_list:
        raise SystemExit("No valid groups found for sequence training")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y, seq_len or 0


def compute_norm_stats(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean/std over all timesteps for feature and target normalization."""
    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    x_mean = X_flat.mean(axis=0, keepdims=True)
    x_std = X_flat.std(axis=0, keepdims=True) + 1e-8
    y_mean = y_flat.mean(axis=0, keepdims=True)
    y_std = y_flat.std(axis=0, keepdims=True) + 1e-8
    return x_mean, x_std, y_mean, y_std


def train_val_split(n_samples: int, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    n_train = int(n_samples * train_ratio)
    return idx[:n_train], idx[n_train:]


# =============== Dataset ===============

class SequencePoseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, stats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        self.X = X.astype("float32")
        self.y = y.astype("float32")
        self.x_mean, self.x_std, self.y_mean, self.y_std = stats
        # reshape means for broadcasting: (1, 1, F)
        self.x_mean = self.x_mean.reshape(1, 1, -1)
        self.x_std = self.x_std.reshape(1, 1, -1)
        self.y_mean = self.y_mean.reshape(1, 1, -1)
        self.y_std = self.y_std.reshape(1, 1, -1)

        self.X_norm = (self.X - self.x_mean) / self.x_std
        self.y_norm = (self.y - self.y_mean) / self.y_std

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X_norm[idx])  # (L, F)
        y = torch.from_numpy(self.y_norm[idx])  # (L, 3)
        return x, y


# =============== Model ===============

class PoseSequenceTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_dim: int = 9,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        out_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, in_dim)
        B, L, _ = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L}")
        h = self.input_proj(x)
        h = h + self.pos_embedding
        h = self.encoder(h)
        out = self.head(h)  # (B, L, out_dim)
        return out


# =============== Metrics ===============

def compute_mae_physical(pred_norm: np.ndarray, gt_norm: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    pred = pred_norm * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
    gt = gt_norm * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
    return np.mean(np.abs(pred - gt), axis=(0, 1))  # (3,)


# =============== Training ===============

def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: amp.GradScaler | None,
    device: torch.device,
    epochs: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    save_path: Path,
    logs_dir: Path,
    save_every: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
):
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    training_logs = []
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start = time.time()
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            if scaler is None:
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
            else:
                with amp.autocast(device_type=device.type):
                    preds = model(x_batch)
                    loss = criterion(preds, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae = None
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)

            val_loss /= len(val_loader.dataset)

            # Compute MAE in physical space on validation set
            all_pred = []
            all_gt = []
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                with torch.no_grad():
                    preds = model(x_batch).cpu().numpy()
                all_pred.append(preds)
                all_gt.append(y_batch.numpy())
            all_pred = np.concatenate(all_pred, axis=0)
            all_gt = np.concatenate(all_gt, axis=0)
            val_mae = compute_mae_physical(all_pred, all_gt, y_mean, y_std)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| val_mae=[{val_mae[0]:.4f}, {val_mae[1]:.4f}, {val_mae[2]:.4f}] | {elapsed:.1f}s"
        )

        training_logs.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mae_x": float(val_mae[0]),
                "val_mae_y": float(val_mae[1]),
                "val_mae_z": float(val_mae[2]),
                "elapsed_sec": float(elapsed),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        # LR scheduler step (plateau)
        if scheduler is not None:
            try:
                scheduler.step(val_loss)
            except Exception as exc:
                print(f"[warn] LR scheduler step failed: {exc}")

        if val_loss < best_val - early_stop_min_delta:
            best_val = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "val_loss": val_loss,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "best_epoch": epoch,
            }
            torch.save(ckpt, save_path)
            print(f"[info] Saved best model to {save_path}")
            early_stop_counter = 0
            best_epoch = epoch
        else:
            early_stop_counter += 1

        if save_every > 0 and epoch % save_every == 0:
            ckpt_path = save_path.parent / f"checkpoint_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"[ckpt] Saved checkpoint: {ckpt_path}")

        # Save running log each epoch to avoid loss on crash
        try:
            import pandas as pd

            pd.DataFrame(training_logs).to_csv(logs_dir / "training_log.csv", index=False)
        except Exception as exc:  # logging should not break training
            print(f"[warn] Failed to write training log: {exc}")

        if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
            print(f"[early-stop] No val improvement for {early_stop_counter} epochs. Stopping early at epoch {epoch}.")
            break

    return best_val, best_epoch if best_epoch is not None else epoch


# =============== Main ===============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sequence transformer mapping end pose sequences to base positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="Path to robot_training_data_seq_expanded.csv")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Directory to save models")
    parser.add_argument("--outputs-dir", type=str, default=str(DEFAULT_OUTPUTS_DIR), help="Directory to save outputs/logs")
    parser.add_argument("--model-name", type=str, default=None, help="Model name (folder). Default uses timestamp.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Sequences per batch")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr-scheduler", choices=["plateau", "none"], default="plateau", help="Learning rate scheduler type")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR decay factor for plateau scheduler")
    parser.add_argument("--lr-patience", type=int, default=10, help="Plateau scheduler patience in epochs")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for plateau scheduler")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio by sequence")
    parser.add_argument("--debug-nrows", type=int, default=None, help="Read only first n rows for quick tests")
    parser.add_argument("--amp", choices=["auto", "true", "false"], default="auto", help="Use mixed precision")
    parser.add_argument("--save-every", type=int, default=0, help="Save extra checkpoints every N epochs (0=off)")
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Stop if val loss does not improve for N epochs (0 disables early stop)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimal improvement on val loss to reset patience")
    return parser.parse_args()


def main() -> None:
    set_seed(42)
    args = parse_args()

    data_path = Path(args.data_path)
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.model_name:
        model_dir = models_dir / args.model_name
    else:
        from datetime import datetime
        model_dir = models_dir / f"seq_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"
    exp_outputs_dir = outputs_dir / model_dir.name
    exp_outputs_dir.mkdir(parents=True, exist_ok=True)

    X, y, seq_len = load_sequences(data_path, nrows=args.debug_nrows)
    n_seq, _, in_dim = X.shape
    print(f"Loaded {n_seq} sequences, seq_len={seq_len}, feature_dim={in_dim}")

    train_idx, val_idx = train_val_split(n_seq, train_ratio=args.train_ratio)
    # Save split for reproducibility
    np.savez(exp_outputs_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx)

    # Compute normalization stats only on train split to avoid data leakage
    x_mean, x_std, y_mean, y_std = compute_norm_stats(X[train_idx], y[train_idx])

    train_ds = SequencePoseDataset(X[train_idx], y[train_idx], (x_mean, x_std, y_mean, y_std))
    val_ds = SequencePoseDataset(X[val_idx], y[val_idx], (x_mean, x_std, y_mean, y_std))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp == "true" or (args.amp == "auto" and device.type == "cuda")

    model = PoseSequenceTransformer(seq_len=seq_len, in_dim=in_dim, out_dim=y.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )
    scaler = amp.GradScaler() if use_amp else None

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    print(f"Device: {device} | AMP: {use_amp} | Save to: {model_path}")
    best_val, best_epoch = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        epochs=args.epochs,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        save_path=model_path,
        logs_dir=exp_outputs_dir,
        save_every=args.save_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    # Save metadata for quick lookup
    meta = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "n_sequences": int(n_seq),
        "seq_len": int(seq_len),
        "train_ratio": float(args.train_ratio),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
    }
    try:
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:
        print(f"[warn] Failed to write metadata: {exc}")


if __name__ == "__main__":
    main()
