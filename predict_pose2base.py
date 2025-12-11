import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# 路径配置：要和训练脚本保持一致
MODEL_PATH = Path("./pose2base_robot_transformer_best.pt")
DATA_PATH = Path("./robot_training_data.csv")
SPLIT_PATH = Path("./split_indices.npz")

# 预测集控制：避免对超大 CSV 全量推理
# PRED_USE_VAL=1 时优先使用 split_indices.npz 中的 val_idx；设为 0 则改为采样
# PRED_SAMPLE_LIMIT 控制最多使用多少行（对 val 集也生效）；默认 2000 行
PRED_USE_VAL = os.environ.get("PRED_USE_VAL", "1")
PRED_SAMPLE_LIMIT = int(os.environ.get("PRED_SAMPLE_LIMIT", "2000"))


# =============== 模型结构（必须和训练时一模一样） ===============

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
            norm_first=True,
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
        """
        x: (B, 9) 标准化后的特征
        """
        B, L = x.shape
        assert L == self.seq_len, f"预期输入长度 {self.seq_len}，但得到 {L}"

        x = x.unsqueeze(-1)            # (B, 9, 1)
        x = self.input_proj(x)         # (B, 9, d_model)
        x = x + self.pos_embedding     # (B, 9, d_model)
        h = self.encoder(x)            # (B, 9, d_model)
        h_pool = h.mean(dim=1)         # (B, d_model)
        out = self.head(h_pool)        # (B, 3)，仍然是标准化空间
        return out


# =============== 加载模型 + 归一化参数 ===============

def load_model(preferred_device: str = "cpu"):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件 {MODEL_PATH}，请先运行训练脚本 train_pose2base_stable.py"
        )

    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    # 和训练时完全相同的结构参数
    model = PoseToBaseTransformer(
        seq_len=9,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        out_dim=3,
        dropout=0.1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_mean = ckpt["x_mean"]  # (1, 9)
    x_std = ckpt["x_std"]    # (1, 9)
    y_mean = ckpt["y_mean"]  # (1, 3)
    y_std = ckpt["y_std"]    # (1, 3)

    # device 选择：默认 cpu；显式设置 PRED_DEVICE=cuda 可尝试 GPU，失败回退 CPU
    if preferred_device == "cuda":
        device = torch.device("cuda")
    elif preferred_device == "cpu":
        device = torch.device("cpu")
    else:  # 容错：未知值一律用 CPU
        device = torch.device("cpu")

    try:
        model.to(device)
    except Exception as exc:  # GPU 配置异常时自动回退 CPU
        print(f"[warn] 将模型放到 {device} 失败，回退到 CPU。原因: {exc}")
        device = torch.device("cpu")
        model.to(device)

    print("=== 已加载模型与归一化参数 ===")
    print("best_val_mse:", ckpt.get("best_val_mse", None))
    print("best_epoch:", ckpt.get("epoch", None))
    print("device:", device)

    return model, device, x_mean, x_std, y_mean, y_std


# =============== 特征构造：必须和训练完全一致 ===============

def build_features_from_raw(poses_raw):
    """
    poses_raw: (N, 6)
      [end_pos_x, end_pos_y, end_pos_z, end_pose_x, end_pose_y, end_pose_z]
      其中姿态角为“度”（和 CSV 一致）

    返回:
      feats: (N, 9) = [pos, sin(angles), cos(angles)]
    """
    poses_raw = np.asarray(poses_raw, dtype="float32")
    pos = poses_raw[:, :3]      # (N, 3)
    ang_deg = poses_raw[:, 3:]  # (N, 3)

    # 角度：度 -> 弧度 -> sin/cos
    ang_rad = np.deg2rad(ang_deg)
    ang_sin = np.sin(ang_rad)
    ang_cos = np.cos(ang_rad)

    feats = np.concatenate([pos, ang_sin, ang_cos], axis=1)  # (N, 9)
    return feats


def predict_from_poses(model, device, x_mean, x_std, y_mean, y_std, poses_raw):
    """
    输入：原始末端位姿数组 (N, 6)
    输出：预测的基座坐标 (N, 3)，单位和 CSV 中一致
    """
    feats = build_features_from_raw(poses_raw)   # (N, 9)
    # 使用训练时保存的均值/方差做标准化
    feats_norm = (feats - x_mean) / x_std        # 广播 (N, 9) - (1, 9)

    feats_tensor = torch.from_numpy(feats_norm).to(device)

    with torch.no_grad():
        preds_norm = model(feats_tensor)         # (N, 3)，标准化空间
        preds_norm = preds_norm.cpu().numpy()

    # 反标准化
    preds = preds_norm * y_std + y_mean          # (N, 3)
    return preds


# =============== Demo 1：手动输入测试 ===============

def demo_manual_input(model, device, x_mean, x_std, y_mean, y_std):
    # 尽量选在训练数据附近的姿态
    poses_raw = [
        [-1460.68, -330.96, 4140.71, 180.0, 41.0, -90.0],
        [-1460.68, -320.68, 4287.66, 180.0, 41.0, -90.0],
    ]

    preds = predict_from_poses(model, device, x_mean, x_std, y_mean, y_std, poses_raw)

    print("===== 手动输入末端位姿 → 预测基座坐标 (stable baseline) =====")
    for pose, xyz in zip(poses_raw, preds):
        print("end  pose [x,y,z, rx,ry,rz]:", pose)
        print("base pos [x,y,z] (pred):     ", [float(f"{v:.3f}") for v in xyz.tolist()])
        print("-" * 60)


# =============== Demo 2：从 CSV 抽样，对比 GT vs Pred + MAE ===============

def demo_from_csv(model, device, x_mean, x_std, y_mean, y_std):
    if not DATA_PATH.exists():
        print(f"找不到 {DATA_PATH}，跳过 CSV 示例")
        return

    df = pd.read_csv(DATA_PATH)

    in_cols = [
        "end_pos_x",
        "end_pos_y",
        "end_pos_z",
        "end_pose_x",
        "end_pose_y",
        "end_pose_z",
    ]
    out_cols = [
        "base_pos_x",
        "base_pos_y",
        "base_pos_z",
    ]

    df = df.dropna(subset=in_cols + out_cols).reset_index(drop=True)

    # 仅使用验证集索引；如未提供则做子样本采样，避免 200w 行全量推理
    df_sample = None
    source = "sample"
    if SPLIT_PATH.exists() and PRED_USE_VAL != "0":
        split = np.load(SPLIT_PATH)
        val_idx = split.get("val_idx", None)
        if val_idx is not None:
            val_idx = np.asarray(val_idx, dtype=np.int64)
            val_idx = val_idx[val_idx < len(df)]
            if PRED_SAMPLE_LIMIT > 0 and len(val_idx) > PRED_SAMPLE_LIMIT:
                val_idx = val_idx[:PRED_SAMPLE_LIMIT]
            if len(val_idx) > 0:
                df_sample = df.loc[val_idx].reset_index(drop=True)
                source = f"val_idx (capped {len(df_sample)})"

    if df_sample is None:
        sample_n = len(df) if PRED_SAMPLE_LIMIT <= 0 else min(PRED_SAMPLE_LIMIT, len(df))
        df_sample = df.sample(n=sample_n, random_state=0).reset_index(drop=True)
        source = f"random_sample {sample_n}"

    poses_raw = df_sample[in_cols].values.astype("float32")  # (M, 6)
    gt_base = df_sample[out_cols].values.astype("float32")   # (M, 3)

    preds = predict_from_poses(model, device, x_mean, x_std, y_mean, y_std, poses_raw)

    print(f"===== 从 CSV 读取样本 → 真实 vs 预测 (stable baseline) | source: {source} | rows: {len(df_sample)} =====")
    for i in range(len(df_sample)):
        print(f"样本 #{i}")
        print("end  pose [x,y,z, rx,ry,rz]:", poses_raw[i].tolist())
        print("base pos GT [x,y,z]:        ", [float(f"{v:.3f}") for v in gt_base[i].tolist()])
        print("base pos Pred [x,y,z]:      ", [float(f"{v:.3f}") for v in preds[i].tolist()])
        print("-" * 80)

    mae = np.mean(np.abs(preds - gt_base), axis=0)
    print("整体 MAE [x, y, z]:", mae.tolist())

def eval_on_val_set(model, device, x_mean, x_std, y_mean, y_std):
    if (not DATA_PATH.exists()) or (not SPLIT_PATH.exists()):
        print("缺少 CSV 或 split_indices.npz，无法在验证集上评估")
        return

    df = pd.read_csv(DATA_PATH)
    in_cols = [
        "end_pos_x", "end_pos_y", "end_pos_z",
        "end_pose_x", "end_pose_y", "end_pose_z",
    ]
    out_cols = ["base_pos_x", "base_pos_y", "base_pos_z"]
    df = df.dropna(subset=in_cols + out_cols).reset_index(drop=True)

    split = np.load(SPLIT_PATH)
    val_idx = split["val_idx"]

    poses_raw = df.loc[val_idx, in_cols].values.astype("float32")
    gt_base   = df.loc[val_idx, out_cols].values.astype("float32")

    preds = predict_from_poses(model, device, x_mean, x_std, y_mean, y_std, poses_raw)

    mae = np.mean(np.abs(preds - gt_base), axis=0)
    print("【验证集】整体 MAE [x, y, z]:", mae.tolist())

def main():
    # 可通过环境变量 PRED_DEVICE 强制 cpu/cuda；默认 cpu 更稳妥
    preferred_device = os.environ.get("PRED_DEVICE", "cpu").lower()
    model, device, x_mean, x_std, y_mean, y_std = load_model(preferred_device)
    demo_manual_input(model, device, x_mean, x_std, y_mean, y_std)
    demo_from_csv(model, device, x_mean, x_std, y_mean, y_std)
    eval_on_val_set(model, device, x_mean, x_std, y_mean, y_std)

if __name__ == "__main__":
    main()