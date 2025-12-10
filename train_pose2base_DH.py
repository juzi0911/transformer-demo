import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import random

DATA_PATH = Path("./robot_training_data.csv")
MODEL_PATH = Path("./pose2base_robot_transformer_best.pt")
SPLIT_PATH = Path("./split_indices.npz")  # 固定 train/val 划分用


# =============== 固定随机种子 ===============
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============== 数据集（pos + sin/cos） ===============
class PoseDataset(Dataset):
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)

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

        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.in_dim = X.shape[1]
        self.out_dim = y.shape[1]

        # 归一化参数
        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True) + 1e-8
        self.y_mean = y.mean(axis=0, keepdims=True)
        self.y_std = y.std(axis=0, keepdims=True) + 1e-8

        self.X_norm = (X - self.x_mean) / self.x_std
        self.y_norm = (y - self.y_mean) / self.y_std

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
        return train_idx, val_idx

    # 第一次：生成并保存
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_train = int(n_samples * train_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    np.savez(SPLIT_PATH, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


# =============== 计算损失函数 ===============
def calculate_base_from_dh(dh_params, end_effector_pose):
    """
    基于 DH 参数计算基座位置
    假设 DH 参数和末端执行器姿态一起输入，返回基座位置
    """
    # 假设计算方法：这里是一个示例，实际计算可以根据你的 DH 模型来实现
    # 这里只返回简单的基座位置示例
    base_x = end_effector_pose[0] + dh_params[0][0]
    base_y = end_effector_pose[1] + dh_params[1][1]
    base_z = end_effector_pose[2] + dh_params[2][2]
    return torch.tensor([base_x, base_y, base_z])


def loss_function(preds, y_batch, dh_params):
    """
    自定义损失函数：包括基座位置预测损失 + DH 参数验证损失
    """
    # 假设 preds 是预测的基座位置 (B, 3)
    predicted_base_pos = preds

    # 计算机械臂的 DH 参数对应的基座位置 (B, 6)
    calculated_base_pos = calculate_base_from_dh(dh_params, y_batch)  # 假设这个函数返回的是一个包含位置和姿态的输出，形状为 (B, 6)

    # 从计算的基座位置中提取 x, y, z 坐标（前3个值）
    calculated_base_pos = calculated_base_pos[:, :3]  # (B, 3)

    # 计算基座位置的 MSE 损失
    base_loss = torch.mean((predicted_base_pos - calculated_base_pos) ** 2)

    return base_loss


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


# =============== 训练函数 ===============
def train(model, train_loader, optimizer, dh_params, device):
    model.train()
    train_loss_sum = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_function(preds, y_batch, dh_params)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * X_batch.size(0)
    train_mse = train_loss_sum / len(train_loader.dataset)
    return train_mse


# =============== 主函数 ===============
def main():
    set_seed(42)

    # 定义 DH 参数（六轴机械臂的 DH 参数）
    dh_params = [
        [0.0, 0.0, 0.0, 0.0],                            # J1
        [250.0 / 1000.0, np.pi / 2, 640.0 / 1000.0, 0.0], # J2
        [630.0 / 1000.0, 0.0, 0.0, np.pi / 2],            # J3
        [-196.0 / 1000.0, -np.pi / 2, 0.0, np.pi],         # J4
        [0.0, np.pi / 2, 1097.5 / 1000.0, 0.0],           # J5
        [0.0, -np.pi / 2, 0.0, 0.0]                       # J6
    ]

    dataset = PoseDataset(DATA_PATH)
    n_samples = len(dataset)

    train_idx, val_idx = get_train_val_indices(n_samples, train_ratio=0.8)

    # Subset 手动切分
    train_X = dataset.X_norm[train_idx]
    train_y = dataset.y_norm[train_idx]
    val_X = dataset.X_norm[val_idx]
    val_y = dataset.y_norm[val_idx]

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(val_X), torch.from_numpy(val_y)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseToBaseTransformer(
        seq_len=dataset.in_dim,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        out_dim=dataset.out_dim,
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 200

    best_val_mse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # ----- train -----        
        train_mse = train(model, train_loader, optimizer, dh_params, device)
        
        # ----- val -----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)
        val_mse = val_loss_sum / len(val_loader.dataset)

        print(f"Epoch {epoch:03d}/{epochs} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {
                "model_state_dict": model.state_dict(),
                "x_mean": dataset.x_mean,
                "x_std": dataset.x_std,
                "y_mean": dataset.y_mean,
                "y_std": dataset.y_std,
                "best_val_mse": best_val_mse,
                "epoch": epoch,
            }

    torch.save(best_state, MODEL_PATH)
    print(f"训练完成，最佳 Val MSE = {best_val_mse:.6f}，模型已保存到: {MODEL_PATH}")


if __name__ == "__main__":
    main()
