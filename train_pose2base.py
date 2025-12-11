import argparse
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import random

DATA_PATH = Path("./data/robot_training_data_expanded.csv")

DEFAULT_MODELS_DIR = Path("./models")
DEFAULT_OUTPUTS_DIR = Path("./outputs")

# python -u train_pose2base.py \
#   --model-name my_experiment \
#   --debug-nrows 10000 \
#   --epochs 2 \
#   --batch-size 128 \
#   --num-workers 2 \
#   --amp auto 

# python -u train_pose2base.py \
#   --model-name full_train \
#   --epochs 200 \
#   --batch-size 256 \
#   --num-workers 8 \
#   --amp auto

# =============== 固定随机种子 ===============

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============== 特征构造 ===============

def load_xy_from_csv(csv_path: Path, nrows: int = None):
    """
    读取 CSV，返回 (X, y)。
    X: [pos, sin(angles), cos(angles)] -> (N, 9)
    y: base_pos -> (N, 3)
    """
    if nrows is None:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, nrows=nrows)

    pos_cols = ["end_pos_x", "end_pos_y", "end_pos_z"]
    ang_cols = ["end_pose_x", "end_pose_y", "end_pose_z"]
    out_cols = ["base_pos_x", "base_pos_y", "base_pos_z"]

    df = df.dropna(subset=pos_cols + ang_cols + out_cols)

    pos = df[pos_cols].values.astype("float32")
    ang_deg = df[ang_cols].values.astype("float32")
    ang_rad = np.deg2rad(ang_deg)
    ang_sin = np.sin(ang_rad)
    ang_cos = np.cos(ang_rad)

    X = np.concatenate([pos, ang_sin, ang_cos], axis=1)  # (N, 9)
    y = df[out_cols].values.astype("float32")            # (N, 3)
    return X, y


# =============== 数据集（pos + sin/cos） ===============

class PoseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, stats):
        """
        X, y: 原始数组（未标准化）
        stats: (x_mean, x_std, y_mean, y_std) —— 应基于训练集计算
        """
        self.X = X.astype("float32")
        self.y = y.astype("float32")
        self.x_mean, self.x_std, self.y_mean, self.y_std = stats

        self.X_norm = (self.X - self.x_mean) / self.x_std
        self.y_norm = (self.y - self.y_mean) / self.y_std

        self.n_samples = self.X.shape[0]
        self.in_dim = self.X.shape[1]
        self.out_dim = self.y.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.X_norm[idx]
        y = self.y_norm[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# =============== 小号 Transformer ===============

class PoseToBaseTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 9,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        out_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.input_proj = nn.Linear(1, d_model)
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

    def _reset_parameters(self):
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        assert L == self.seq_len
        x = x.unsqueeze(-1)            # (B, 9, 1)
        x = self.input_proj(x)         # (B, 9, d_model)
        x = x + self.pos_embedding
        h = self.encoder(x)            # (B, 9, d_model)
        h_pool = h.mean(dim=1)         # (B, d_model)
        out = self.head(h_pool)        # (B, 3)
        return out


# =============== 固定 train/val 划分 ===============

def get_train_val_indices(n_samples: int, train_ratio: float = 0.8):
    if SPLIT_PATH.exists():
        data = np.load(SPLIT_PATH)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        # 如果已保存的索引不匹配当前样本数（例如调试只读前几行），则重新生成划分
        if len(train_idx) == n_samples:
            return train_idx, val_idx
        else:
            print("Saved split does not match current dataset size -> regenerating split")

    # 第一次：生成并保存
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_train = int(n_samples * train_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    np.savez(SPLIT_PATH, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


def compute_mae_physical(model, loader, device, y_mean, y_std):
    model.eval()
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds_norm = model(X_batch)
            all_pred.append(preds_norm.cpu().numpy())
            all_gt.append(y_batch.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    # 反归一化到物理坐标
    all_pred_phys = all_pred * y_std + y_mean
    all_gt_phys   = all_gt   * y_std + y_mean

    mae = np.mean(np.abs(all_pred_phys - all_gt_phys), axis=0)
    return mae


def main():
    set_seed(42)
    # 声明将在本函数修改的全局路径变量，必须在首次使用这些名字之前声明
    global DATA_PATH, MODEL_PATH, SPLIT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-nrows", type=int, default=None,
                        help="只读取前 n 行用于快速调试")
    parser.add_argument("--amp", choices=["auto", "true", "false"], default="auto",
                        help="是否使用混合精度: auto/true/false（auto 在有 CUDA 时启用）")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH),
                        help="训练数据 CSV 文件路径")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR),
                        help="保存模型的目录")
    parser.add_argument("--outputs-dir", type=str, default=str(DEFAULT_OUTPUTS_DIR),
                        help="保存 split 等产物的目录")
    parser.add_argument("--model-name", type=str, default=None,
                        help="模型文件名（例如 mymodel.pt）。如果为空则使用默认名")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 路径恢复训练")
    parser.add_argument("--save-every", type=int, default=0,
                        help="每隔 N 个 epoch 额外保存一次 checkpoint（0 表示仅保存最佳模型）")
    parser.add_argument("--lr-scheduler", choices=["plateau", "none"], default="plateau",
                        help="学习率调度器类型：plateau 或 none")
    parser.add_argument("--lr-factor", type=float, default=0.5,
                        help="ReduceLROnPlateau 的衰减因子")
    parser.add_argument("--lr-patience", type=int, default=10,
                        help="ReduceLROnPlateau 的等待 epoch 数")
    parser.add_argument("--lr-min", type=float, default=1e-6,
                        help="ReduceLROnPlateau 的最小学习率")
    parser.add_argument("--early-stop-patience", type=int, default=20,
                        help="早停等待的 epoch 数 (val 指标未改善时累积)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                        help="早停判定的最小改善幅度 (val_mse 需至少降低该值)")
    args = parser.parse_args()

    # 根据传入参数设置路径并确保目录存在
    DATA_PATH = Path(args.data_path)
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # model name: 如果用户没提供，使用带时间戳的默认名，避免覆盖
    if args.model_name:
        model_name = args.model_name
        if not model_name.endswith('.pt'):
            model_name = model_name + '.pt'
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"pose2base_{ts}.pt"

    # 将同一实验的 artifacts 放到以 model_name 为名的子文件夹中，便于管理多个实验
    model_stem = model_name[:-3] if model_name.endswith('.pt') else model_name
    exp_models_dir = models_dir / model_stem
    exp_outputs_dir = outputs_dir / model_stem
    exp_models_dir.mkdir(parents=True, exist_ok=True)
    exp_outputs_dir.mkdir(parents=True, exist_ok=True)

    # Use fixed model filename inside experiment folder to simplify references
    MODEL_PATH = exp_models_dir / "model.pt"
    SPLIT_PATH = exp_outputs_dir / "split_indices.npz"

    # model tag: 标识是 full 训练还是 debug_nrows（提前定义，便于日志记录）
    model_tag = f"debug_nrows={args.debug_nrows}" if args.debug_nrows is not None else "trained_on=full"

    # 读取原始特征 & 标签
    X_all, y_all = load_xy_from_csv(DATA_PATH, nrows=args.debug_nrows)
    n_samples = len(X_all)

    train_idx, val_idx = get_train_val_indices(n_samples, train_ratio=0.8)

    # 仅用训练集计算归一化参数，避免数据泄露
    x_mean = X_all[train_idx].mean(axis=0, keepdims=True)
    x_std = X_all[train_idx].std(axis=0, keepdims=True) + 1e-8
    y_mean = y_all[train_idx].mean(axis=0, keepdims=True)
    y_std = y_all[train_idx].std(axis=0, keepdims=True) + 1e-8
    train_stats = (x_mean, x_std, y_mean, y_std)

    # 如果 resume，优先使用 checkpoint 中的归一化参数，确保一致
    ckpt = None
    start_epoch = 1

    # 提前准备数据集，但在确认最终 stats（若 resume 覆盖）后再创建
    # 此处先不创建 dataset，等 resume 检查完成

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型
    model = PoseToBaseTransformer(
        seq_len=X_all.shape[1],
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        out_dim=y_all.shape[1],
        dropout=0.1,
    ).to(device)

    # 决定是否使用 AMP：仅在有 CUDA 时启用混合精度
    if args.amp == "true":
        use_amp = (device.type == "cuda")
    elif args.amp == "false":
        use_amp = False
    else:
        use_amp = (device.type == "cuda")

    scaler = amp.GradScaler() if use_amp else None

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = args.epochs
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )

    # ====== 可选恢复 ======
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"[info] 从 checkpoint 恢复: {resume_path}")
            ckpt = torch.load(resume_path, map_location="cpu")
            # 如果 checkpoint 保存了归一化参数，覆盖本次计算的 stats，确保一致性
            if all(k in ckpt for k in ["x_mean", "x_std", "y_mean", "y_std"]):
                train_stats = (
                    ckpt["x_mean"],
                    ckpt["x_std"],
                    ckpt["y_mean"],
                    ckpt["y_std"],
                )
            # 加载模型 / 优化器 / scaler 状态
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scaler is not None and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
        else:
            print(f"[warn] 未找到 checkpoint: {resume_path}，将从头训练")

    # 基于最终的 train_stats 创建标准化后的 Dataset / Loader
    x_mean, x_std, y_mean, y_std = train_stats
    train_ds = PoseDataset(X_all[train_idx], y_all[train_idx], train_stats)
    val_ds = PoseDataset(X_all[val_idx], y_all[val_idx], train_stats)

    use_cuda = (device.type == "cuda")
    pin_memory = use_cuda

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    # 根据当前硬件给出建议（仅打印）
    cpu_count = None
    try:
        import os
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1

    suggested_num_workers = min(8, max(0, cpu_count // 2))
    suggested_batch = 256 if torch.cuda.is_available() else 64
    print("Training suggestions:")
    print(f"  - Detected device: {device}")
    print(f"  - Suggested batch_size: {suggested_batch} (adjust for your GPU memory)")
    print(f"  - Suggested num_workers: {suggested_num_workers}")
    print(f"  - Mixed precision (AMP) suggested: {use_amp}")

    best_val_mse = float("inf")
    best_state = None
    if ckpt is not None:
        best_val_mse = float(ckpt.get("best_val_mse", best_val_mse))
        best_state = ckpt.get("best_state", best_state)
    early_stop_counter = 0
    # 用于保存每个 epoch 的训练日志（稍后写入 outputs_dir/training_log.csv ）
    training_logs = []

    for epoch in range(start_epoch, epochs + 1):
        # ----- train -----
        model.train()
        train_loss_sum = 0.0
        processed = 0
        epoch_start = time.time()

        train_iter = train_loader
        train_bar = tqdm(train_iter, desc=f"Epoch {epoch} Train", leave=False)
        for X_batch, y_batch in train_bar:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with amp.autocast(device_type=device.type):
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

            bsize = X_batch.size(0)
            train_loss_sum += loss.item() * bsize
            processed += bsize
            elapsed = time.time() - epoch_start
            samples_per_sec = processed / elapsed if elapsed > 0 else 0.0
            train_bar.set_postfix(loss=loss.item(), samples_per_sec=f"{samples_per_sec:.1f}")
        train_mse = train_loss_sum / len(train_ds)

        # ----- val -----
        model.eval()
        val_loss_sum = 0.0
        val_processed = 0
        val_epoch_start = time.time()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False)
        with torch.no_grad():
            for X_batch, y_batch in val_bar:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                if use_amp:
                    with amp.autocast(device_type=device.type):
                        preds = model(X_batch)
                        loss = criterion(preds, y_batch)
                else:
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                bsize = X_batch.size(0)
                val_loss_sum += loss.item() * bsize
                val_processed += bsize
                velapsed = time.time() - val_epoch_start
                v_sps = val_processed / velapsed if velapsed > 0 else 0.0
                val_bar.set_postfix(loss=loss.item(), samples_per_sec=f"{v_sps:.1f}")
        val_mse = val_loss_sum / len(val_ds)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}"
        )

        # 调度学习率
        if scheduler is not None:
            scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            # model tag: 标识是 full 训练还是 debug_nrows
            model_tag = f"debug_nrows={args.debug_nrows}" if args.debug_nrows is not None else "trained_on=full"
            best_state = {
                "model_state_dict": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "best_val_mse": best_val_mse,
                "epoch": epoch,
                "n_samples": n_samples,
                "model_tag": model_tag,
            }
            early_stop_counter = 0
        else:
            # 未提升则计数
            if val_mse > best_val_mse - args.early_stop_min_delta:
                early_stop_counter += 1
            else:
                early_stop_counter = 0

        train_mae = compute_mae_physical(model, train_loader, device, y_mean, y_std)
        val_mae   = compute_mae_physical(model, val_loader,   device, y_mean, y_std)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}: Train MAE {train_mae}, Val MAE {val_mae}")

        # 早停检查
        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(f"[early-stop] 连续 {early_stop_counter} 个 epoch 验证集未改善，提前停止。")
            break

        # 额外 checkpoint（可选）
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = exp_models_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_val_mse": best_val_mse,
                "best_state": best_state,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "n_samples": n_samples,
                "model_tag": model_tag,
            }, ckpt_path)
            print(f"[ckpt] 保存 checkpoint: {ckpt_path}")

        # 记录日志行（扁平化 MAE 数组）
        # 当前学习率（可能由调度器变化）
        try:
            current_lr = float(optimizer.param_groups[0]['lr'])
        except Exception:
            current_lr = None

        training_logs.append({
            "epoch": epoch,
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "lr": current_lr,
            "model_tag": model_tag,
            "n_samples": int(n_samples),
            "train_mae_x": float(train_mae[0]),
            "train_mae_y": float(train_mae[1]),
            "train_mae_z": float(train_mae[2]),
            "val_mae_x": float(val_mae[0]),
            "val_mae_y": float(val_mae[1]),
            "val_mae_z": float(val_mae[2]),
            "epoch_seconds": float(epoch_time),
            "best_val_mse_so_far": float(best_val_mse),
        })

    if best_state is None:
        # 如果训练未找到更优值，也至少保存最后状态
        best_state = {
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "best_val_mse": best_val_mse,
            "epoch": epochs,
            "n_samples": n_samples,
            "model_tag": model_tag,
        }

    torch.save(best_state, MODEL_PATH)
    # 保存训练日志到 outputs 目录，便于后续可视化
    try:
        logs_df = pd.DataFrame(training_logs)
        # save logs to both outputs and models experiment folders
        logs_path_out = exp_outputs_dir / "training_log.csv"
        logs_path_model = exp_models_dir / "training_log.csv"
        logs_df.to_csv(logs_path_out, index=False)
        logs_df.to_csv(logs_path_model, index=False)
        print(f"训练日志已保存: {logs_path_out} 和 {logs_path_model}")
        # save metadata json in models experiment folder
        meta = {
            "model_name": model_name,
            "model_tag": model_tag,
            "n_samples": int(n_samples),
            "best_val_mse": float(best_val_mse),
            "epoch": int(best_state.get('epoch', -1)) if best_state else None,
        }
        try:
            import json
            meta_path = exp_models_dir / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"模型元数据已保存: {meta_path}")
        except Exception as exc:
            print(f"无法保存模型元数据: {exc}")
    except Exception as exc:
        print(f"无法保存训练日志: {exc}")
    print(f"训练完成，最佳 Val MSE = {best_val_mse:.6f}，模型已保存到: {MODEL_PATH}")


if __name__ == "__main__":
    main()