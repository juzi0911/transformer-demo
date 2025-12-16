#!/usr/bin/env python3
"""Predict base position sequences from end-effector pose sequences using a trained seq model.

Usage example (project root):
  python -u predict_pose2base_sequence.py \
    --model-name seq_exp \
    --max-groups 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEFAULT_DATA_PATH = Path("./data/robot_training_data_seq_expanded.csv")
DEFAULT_MODELS_DIR = Path("./models")
DEFAULT_OUTPUTS_DIR = Path("./outputs")
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "seq_exp" / "model.pt"

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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

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
        out = self.head(h)
        return out


# =============== Data helpers ===============

def build_features_from_df(df: pd.DataFrame) -> np.ndarray:
    pos = df[["end_pos_x", "end_pos_y", "end_pos_z"]].to_numpy(dtype="float32")
    ang_deg = df[["end_pose_x", "end_pose_y", "end_pose_z"]].to_numpy(dtype="float32")
    ang_rad = np.deg2rad(ang_deg)
    ang_sin = np.sin(ang_rad)
    ang_cos = np.cos(ang_rad)
    return np.concatenate([pos, ang_sin, ang_cos], axis=1)


def load_sequences_for_infer(
    csv_path: Path,
    expected_seq_len: int | None = None,
    max_groups: int | None = None,
) -> Tuple[np.ndarray, List[int], List[np.ndarray], List[np.ndarray]]:
    df = pd.read_csv(csv_path)
    required = set(INPUT_COLS + TARGET_COLS + GROUP_COLS)
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=INPUT_COLS + TARGET_COLS + GROUP_COLS)
    df = df.sort_values(["group_id", "point_index"]).reset_index(drop=True)

    seq_len = expected_seq_len

    grouped = [(int(gid), g.reset_index(drop=True)) for gid, g in df.groupby("group_id")]

    X_list: List[np.ndarray] = []
    base_list: List[np.ndarray] = []
    gid_list: List[int] = []
    pidx_list: List[np.ndarray] = []

    for idx, (gid, g) in enumerate(grouped):
        if max_groups is not None and len(X_list) >= max_groups:
            break
        if seq_len is None:
            seq_len = len(g)
        if len(g) != seq_len:
            print(f"[warn] Skip group {gid}: length {len(g)} != expected {seq_len}")
            continue
        feats = build_features_from_df(g)
        base = g[TARGET_COLS].to_numpy(dtype="float32")
        X_list.append(feats)
        base_list.append(base)
        gid_list.append(gid)
        pidx_list.append(g["point_index"].to_numpy())

    if not X_list:
        raise SystemExit("No valid groups found for prediction")

    X = np.stack(X_list, axis=0)
    return X, gid_list, pidx_list, base_list


def compute_feature_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    x_mean = flat.mean(axis=0, keepdims=True)
    x_std = flat.std(axis=0, keepdims=True) + 1e-8
    return x_mean, x_std


# =============== Inference ===============

def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    seq_len = state["pos_embedding"].shape[1]
    d_model = state["pos_embedding"].shape[2]
    in_dim = state["input_proj.weight"].shape[1]
    out_dim = state["head.3.weight"].shape[0]

    model = PoseSequenceTransformer(
        seq_len=seq_len,
        in_dim=in_dim,
        d_model=d_model,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        out_dim=out_dim,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    x_mean = ckpt.get("x_mean")
    x_std = ckpt.get("x_std")
    y_mean = ckpt.get("y_mean")
    y_std = ckpt.get("y_std")
    if y_mean is None or y_std is None:
        raise SystemExit("Checkpoint missing y_mean / y_std; cannot de-normalize outputs.")

    return model, x_mean, x_std, y_mean, y_std, seq_len


def predict_sequences(
    model: nn.Module,
    device: torch.device,
    X: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    X_norm = (X - x_mean) / x_std
    x_tensor = torch.from_numpy(X_norm).to(device)
    with torch.no_grad():
        preds_norm = model(x_tensor).cpu().numpy()
    preds = preds_norm * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
    return preds


# =============== CLI ===============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict base position sequences from end-effector sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--model-name", type=str, default=None, help="Experiment folder name under --models-dir")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Directory containing experiment folders")
    parser.add_argument("--outputs-dir", type=str, default=str(DEFAULT_OUTPUTS_DIR), help="Directory containing experiment outputs (for split_indices.npz)")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="CSV with grouped sequence data")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Inference device")
    parser.add_argument("--max-groups", type=int, default=None, help="Optionally cap number of groups for quick tests")
    parser.add_argument("--expected-seq-len", type=int, default=None, help="Override expected sequence length; otherwise inferred from model or data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)

    # Resolve model path by model-name if provided
    model_path = Path(args.model)
    split_path = None
    if args.model_name:
        exp = Path(args.model_name)
        candidate = None
        if exp.is_dir() and (exp / "model.pt").exists():
            candidate = exp / "model.pt"
        else:
            maybe = models_dir / exp.name / "model.pt"
            if maybe.exists():
                candidate = maybe
        if candidate is not None:
            model_path = candidate
        split_candidate = outputs_dir / exp.name / "split_indices.npz"
        if split_candidate.exists():
            split_path = split_candidate

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    model, x_mean_ckpt, x_std_ckpt, y_mean, y_std, model_seq_len = load_model(model_path, device)
    expected_seq_len = args.expected_seq_len or model_seq_len

    # If split_indices.npz is found, default to using val_idx subset
    seq_indices = None
    if split_path and split_path.exists():
        split = np.load(split_path)
        if "val_idx" in split:
            seq_indices = split["val_idx"]
            print(f"[info] Using val_idx from {split_path}, count={len(seq_indices)}")
        else:
            print(f"[warn] split file {split_path} missing val_idx; using all sequences")

    # Load sequences and optionally filter by val indices
    X_all, gid_all, pidx_all, base_all = load_sequences_for_infer(
        data_path, expected_seq_len=expected_seq_len, max_groups=None
    )
    if seq_indices is not None:
        seq_indices = seq_indices[seq_indices < len(X_all)]
        X = X_all[seq_indices]
        gid_list = [gid_all[i] for i in seq_indices]
        pidx_list = [pidx_all[i] for i in seq_indices]
        base_list = [base_all[i] for i in seq_indices]
    else:
        X, gid_list, pidx_list, base_list = X_all, gid_all, pidx_all, base_all

    if args.max_groups is not None:
        X = X[: args.max_groups]
        gid_list = gid_list[: args.max_groups]
        pidx_list = pidx_list[: args.max_groups]
        base_list = base_list[: args.max_groups]
    # Prefer ckpt feature stats; fallback to recompute from data if missing
    if x_mean_ckpt is None or x_std_ckpt is None:
        x_mean, x_std = compute_feature_stats(X)
    else:
        x_mean, x_std = x_mean_ckpt, x_std_ckpt

    preds = predict_sequences(model, device, X, x_mean, x_std, y_mean, y_std)

    # Print predictions vs GT to terminal
    print(
        f"=== Prediction results | model={model_path} | device={device} | seq_len={expected_seq_len} | groups={len(gid_list)} | split={'val' if seq_indices is not None else 'all'} ==="
    )
    all_abs_err = []
    for i, gid in enumerate(gid_list):
        print(f"-- group {gid} ({len(pidx_list[i])} steps) --")
        for t, pidx in enumerate(pidx_list[i]):
            gt = base_list[i][t]
            pd_ = preds[i, t]
            err = np.abs(pd_ - gt)
            all_abs_err.append(err)
            print(
                f"idx={int(pidx):03d} | GT [x,y,z]=({gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}) "
                f"| Pred=({pd_[0]:.3f}, {pd_[1]:.3f}, {pd_[2]:.3f}) "
                f"| AbsErr=({err[0]:.3f}, {err[1]:.3f}, {err[2]:.3f})"
            )
    if all_abs_err:
        mae = np.mean(np.stack(all_abs_err, axis=0), axis=0)
        print(f"Overall MAE [x,y,z]=({mae[0]:.4f}, {mae[1]:.4f}, {mae[2]:.4f})")


if __name__ == "__main__":
    main()
