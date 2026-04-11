import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # 无图形界面后端，只保存图片
import matplotlib.pyplot as plt

# 如果你装了 sklearn，可以开启随机森林对比（可选传统基线）
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    print("Warning: sklearn not found, RandomForest baseline will be skipped.")


# ===================== 1. 读入数据 =====================

def load_raman_nmr_data(raman_file, nmr_file):
    """
    读取拉曼 + 核磁 Excel
    - 第一行（index=0）为无用信息：删除
    - 第 0 列为中文类别标签（成熟 / 非成熟 / 加热）
    - 其余列为光谱点
    返回：
        X_raman: (N, d_r)
        X_nmr:   (N, d_n)
        y:       (N,)  int 标签
        label_map: {int -> str}
    """
    # 不使用表头，用行列号
    raman_raw = pd.read_excel(raman_file, header=None)
    nmr_raw = pd.read_excel(nmr_file, header=None)

    # 去掉第一行
    raman_data = raman_raw.iloc[1:, :]
    nmr_data = nmr_raw.iloc[1:, :]

    # 标签在第 0 列（中文）
    labels_r = raman_data.iloc[:, 0].astype(str).values
    labels_n = nmr_data.iloc[:, 0].astype(str).values

    # 简单 sanity check：两个表的标签顺序应一致
    if not np.array_equal(labels_r, labels_n):
        print("Warning: Raman & NMR labels differ, using Raman labels.")

    label_str = labels_r
    unique_labels = sorted(list(set(label_str)))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    y = np.array([label_to_idx[s] for s in label_str], dtype=np.int64)

    # 特征：从第 1 列开始
    X_raman = raman_data.iloc[:, 1:].astype(np.float32).values
    X_nmr = nmr_data.iloc[:, 1:].astype(np.float32).values

    return X_raman, X_nmr, y, idx_to_label


# ===================== 2. 数据集 & 客户端划分 =====================

class HoneySpectraDataset(Dataset):
    def __init__(self, X_raman, X_nmr, y, indices,
                 has_raman=True, has_nmr=True):
        self.X_raman = X_raman
        self.X_nmr = X_nmr
        self.y = y
        self.indices = list(indices)
        self.has_raman = has_raman
        self.has_nmr = has_nmr

        self.d_r = X_raman.shape[1]
        self.d_n = X_nmr.shape[1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        if self.has_raman:
            xr = self.X_raman[idx]
        else:
            xr = np.zeros(self.d_r, dtype=np.float32)

        if self.has_nmr:
            xn = self.X_nmr[idx]
        else:
            xn = np.zeros(self.d_n, dtype=np.float32)

        y = self.y[idx]
        return (
            torch.from_numpy(xr),
            torch.from_numpy(xn),
            torch.tensor(y, dtype=torch.long)
        )


def split_train_test(n_samples, test_ratio=0.2, seed=42, y=None):
    """
    简单划分训练 / 测试，支持按 y 分层（可选）。
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)

    if y is None:
        rng.shuffle(indices)
        split = int(n_samples * (1 - test_ratio))
        return indices[:split], indices[split:]
    else:
        # 分层划分
        train_idx = []
        test_idx = []
        labels = np.unique(y)
        for lab in labels:
            lab_idx = np.where(y == lab)[0]
            rng.shuffle(lab_idx)
            split = int(len(lab_idx) * (1 - test_ratio))
            train_idx.extend(lab_idx[:split])
            test_idx.extend(lab_idx[split:])
        return np.array(train_idx), np.array(test_idx)


def create_federated_clients(train_indices, n_clients=3, seed=0):
    """
    模拟不同实验室：
    client 0: 只有拉曼
    client 1: 只有核磁
    client 2: 拉曼 + 核磁
    如果更多客户端，可自行扩展
    """
    rng = np.random.default_rng(seed)
    rng.shuffle(train_indices)

    n = len(train_indices)
    if n_clients < 3:
        raise ValueError("For this demo we assume at least 3 clients.")

    # 简单按 3 份切，剩下的都给最后一个 client
    size = n // 3
    client_indices = []
    client_indices.append(train_indices[:size])            # 0
    client_indices.append(train_indices[size:2 * size])   # 1
    client_indices.append(train_indices[2 * size:])       # 2 (可能略大)

    clients = []
    # client 0: Raman only
    clients.append(dict(indices=client_indices[0],
                        has_raman=True, has_nmr=False))
    # client 1: NMR only
    clients.append(dict(indices=client_indices[1],
                        has_raman=False, has_nmr=True))
    # client 2: Both
    clients.append(dict(indices=client_indices[2],
                        has_raman=True, has_nmr=True))

    return clients


# ===================== 3. 多种模型定义 =====================

class MultiModalMLP(nn.Module):
    """
    双分支 MLP：Raman 编码 + NMR 编码，bottleneck 拼接后分类
    """
    def __init__(self, dim_raman, dim_nmr, n_classes,
                 hidden_enc=256, bottleneck=64, hidden_fusion=128):
        super().__init__()
        self.enc_raman = nn.Sequential(
            nn.Linear(dim_raman, hidden_enc),
            nn.ReLU(),
            nn.Linear(hidden_enc, bottleneck),
            nn.ReLU(),
        )
        self.enc_nmr = nn.Sequential(
            nn.Linear(dim_nmr, hidden_enc),
            nn.ReLU(),
            nn.Linear(hidden_enc, bottleneck),
            nn.ReLU(),
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(2 * bottleneck, hidden_fusion),
            nn.ReLU(),
            nn.Linear(hidden_fusion, n_classes),
        )

    def forward(self, x_raman, x_nmr):
        h_r = self.enc_raman(x_raman)
        h_n = self.enc_nmr(x_nmr)
        h = torch.cat([h_r, h_n], dim=1)
        logits = self.fusion_net(h)
        return logits


class EarlyFusionMLP(nn.Module):
    """
    早期融合 MLP：直接拼接原始 Raman + NMR 特征，再用一个 MLP 分类
    """
    def __init__(self, dim_raman, dim_nmr, n_classes,
                 hidden1=256, hidden2=128):
        super().__init__()
        in_dim = dim_raman + dim_nmr
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x_raman, x_nmr):
        x = torch.cat([x_raman, x_nmr], dim=1)
        logits = self.net(x)
        return logits


class MultiModalCNN1D(nn.Module):
    """
    多模态 1D-CNN：把光谱当成 1D 序列，分别用卷积编码 Raman 和 NMR，再融合分类
    """
    def __init__(self, dim_raman, dim_nmr, n_classes,
                 ch1=16, ch2=32, fc_hidden=64):
        super().__init__()

        # Raman 分支
        self.conv_r1 = nn.Conv1d(1, ch1, kernel_size=5, padding=2)
        self.conv_r2 = nn.Conv1d(ch1, ch2, kernel_size=5, padding=2)
        # NMR 分支
        self.conv_n1 = nn.Conv1d(1, ch1, kernel_size=5, padding=2)
        self.conv_n2 = nn.Conv1d(ch1, ch2, kernel_size=5, padding=2)

        # 全局平均池化到 1 个时间步
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 每个模态输出 ch2 维，全连接部分输入 2*ch2
        self.fc = nn.Sequential(
            nn.Linear(2 * ch2, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_classes),
        )

    def encode_branch(self, x, conv1, conv2):
        # x: [B, L] -> [B, 1, L]
        x = x.unsqueeze(1)
        x = F.relu(conv1(x))
        x = F.relu(conv2(x))
        x = self.pool(x)          # [B, C, 1]
        x = x.view(x.size(0), -1) # [B, C]
        return x

    def forward(self, x_raman, x_nmr):
        h_r = self.encode_branch(x_raman, self.conv_r1, self.conv_r2)
        h_n = self.encode_branch(x_nmr, self.conv_n1, self.conv_n2)
        h = torch.cat([h_r, h_n], dim=1)
        logits = self.fc(h)
        return logits


# ===================== 4. 训练 & 评估工具函数 =====================

def train_local(model, dataloader, epochs, device, lr=1e-3):
    """联邦场景下的本地训练（返回 state_dict）"""
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for xr, xn, y in dataloader:
            xr = xr.to(device)
            xn = xn.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(xr, xn)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def train_centralized(model, dataloader, epochs, device, lr=1e-3):
    """集中式训练（直接返回训练好的模型）"""
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for xr, xn, y in dataloader:
            xr = xr.to(device)
            xn = xn.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(xr, xn)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, dataloader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xr, xn, y in dataloader:
            xr = xr.to(device)
            xn = xn.to(device)
            y = y.to(device)
            logits = model(xr, xn)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def fedavg(state_dicts, weights=None):
    """
    简单 FedAvg：对 state_dict 逐元素加权平均
    state_dicts: list of OrderedDict
    weights: list of float，长度同 state_dicts；若 None 则均匀平均
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        s = sum(weights)
        weights = [w / s for w in weights]

    # 初始化
    avg_state = {}
    for k in state_dicts[0].keys():
        avg_state[k] = sum(
            w * state_dicts[i][k] for i, w in enumerate(weights)
        )
    return avg_state


# ===================== 5. 压缩函数：随机稀疏 + 随机量化 =====================

def compress_state_dict(delta_sd, scheme="none", sparsity=0.1, num_bits=8):
    """
    对本地更新 delta_sd 做压缩：
      - scheme="none"：不压缩，视为 32bit 浮点
      - scheme="q_sparse"：随机保留一定比例的元素 + 统一标度量化到 num_bits 比特
    返回：
      - compressed_sd: 压缩后的（已解压回 float）delta，用于 FedAvg
      - total_bits: 本轮该 client 需要传输的比特数（粗略估计）
    """
    compressed = {}
    total_params = 0
    total_nonzero = 0

    for k, v in delta_sd.items():
        if not isinstance(v, torch.Tensor):
            compressed[k] = v
            continue

        tensor = v.detach().clone()
        numel = tensor.numel()
        total_params += numel

        if scheme == "none":
            compressed[k] = tensor
            total_nonzero += numel
        else:
            # 随机稀疏
            flat = tensor.view(-1)
            numel = flat.numel()
            k_keep = max(1, int(numel * sparsity))
            idx = torch.randperm(numel, device=flat.device)[:k_keep]
            mask = torch.zeros(numel, dtype=torch.bool, device=flat.device)
            mask[idx] = True

            flat_compressed = torch.zeros_like(flat)
            kept_vals = flat[mask]

            if kept_vals.numel() > 0:
                max_abs = kept_vals.abs().max()
                if max_abs > 0:
                    levels = 2 ** num_bits
                    step = 2 * max_abs / (levels - 1)
                    q = torch.clamp(torch.round(kept_vals / step),
                                    -(levels // 2),
                                    levels // 2 - 1)
                    kept_vals_q = q * step
                else:
                    kept_vals_q = kept_vals
                flat_compressed[mask] = kept_vals_q

            tensor_compressed = flat_compressed.view_as(tensor)
            compressed[k] = tensor_compressed
            total_nonzero += k_keep

    if scheme == "none":
        bits_per_param = 32.0
        total_bits = total_params * bits_per_param
    else:
        # 为简单起见，忽略索引比特，只按稀疏比例乘 num_bits
        bits_per_param_effective = sparsity * num_bits
        total_bits = total_params * bits_per_param_effective

    return compressed, float(total_bits)


# ===================== 6. RandomForest（集中式传统对比，可选） =====================

def run_random_forest_baseline(X_raman, X_nmr, y, train_idx, test_idx):
    if not HAVE_SKLEARN:
        print("sklearn not installed, skip RandomForest baseline.")
        return

    # 简单早期融合：直接拼接特征
    X = np.concatenate([X_raman, X_nmr], axis=1)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=0
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Centralized RandomForest] Test accuracy = {acc:.4f}")


# ===================== 7. 联邦实验（给定模型） =====================

def run_federated_experiment(model_name, model_ctor,
                             client_loaders, client_weights,
                             test_loader, device,
                             num_rounds=20, local_epochs=3, lr=1e-3,
                             compress=False, sparsity=0.1, num_bits=8,
                             comm_bits_dict=None):
    """
    compress=False：普通 FedAvg
    compress=True：对本地更新做随机稀疏 + 随机量化，然后再 FedAvg
    通信量按 “上行 + 下行” 统计：
      - 下行：每轮服务器广播一次全局模型
      - 上行：每个客户端上传其本地更新（压缩/未压缩）
    """
    print(f"\n===== Start Federated Training: {model_name} "
          f"(compress={compress}) =====")

    # 初始化全局模型
    global_model = model_ctor()
    global_state = global_model.state_dict()

    # 参数总数，用于估算下行通信
    n_params = sum(p.numel() for p in global_model.parameters())

    acc_history = []
    total_bits_all_rounds = 0.0

    scheme = "q_sparse" if compress else "none"

    for rnd in range(1, num_rounds + 1):
        local_states_for_agg = []

        # -------- 下行通信：广播全局模型 --------
        if scheme == "none":
            bits_down = n_params * 32.0          # 32-bit float
        else:
            bits_down = n_params * sparsity * num_bits
        total_bits_all_rounds += bits_down

        # -------- 上行通信：各客户端上传本地更新 --------
        for cid, ((dl, has_r, has_n), w) in enumerate(
                zip(client_loaders, client_weights)):
            # 每个客户端从全局模型状态开始本地训练
            local_model = model_ctor()
            local_model.load_state_dict(global_state)

            sd_local = train_local(local_model, dl, local_epochs, device, lr)

            # 计算本地更新 delta = local - global
            delta_sd = {}
            for k in global_state.keys():
                delta_sd[k] = sd_local[k] - global_state[k]

            # 压缩本地更新
            compressed_delta, bits_up = compress_state_dict(
                delta_sd, scheme=scheme, sparsity=sparsity, num_bits=num_bits
            )
            total_bits_all_rounds += bits_up

            # 还原出“压缩后的本地模型”（server 端解压后的效果）
            approx_local_state = {}
            for k in global_state.keys():
                approx_local_state[k] = global_state[k] + compressed_delta[k]

            local_states_for_agg.append(approx_local_state)

        # FedAvg 聚合
        global_state = fedavg(local_states_for_agg, weights=client_weights)
        global_model.load_state_dict(global_state)

        # 每轮在测试集上评估
        test_acc = evaluate(global_model, test_loader, device)
        acc_history.append(test_acc)
        print(f"[{model_name}] Round {rnd:02d} - Test accuracy = {test_acc:.4f}")

    print(f"===== {model_name} Finished. "
          f"Best test acc = {max(acc_history):.4f} =====")

    if comm_bits_dict is not None:
        comm_bits_dict[model_name] = total_bits_all_rounds

    return acc_history


# ===================== 8. 集中式实验（给定模型） =====================

def run_centralized_experiment(model_name, model_ctor,
                               X_raman, X_nmr, y,
                               train_idx, test_idx,
                               device,
                               epochs=100, batch_size=16, lr=1e-3):
    print(f"\n===== Start Centralized Training: {model_name} =====")

    train_dataset = HoneySpectraDataset(
        X_raman, X_nmr, y, indices=train_idx,
        has_raman=True, has_nmr=True
    )
    test_dataset = HoneySpectraDataset(
        X_raman, X_nmr, y, indices=test_idx,
        has_raman=True, has_nmr=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model_ctor()
    model = train_centralized(model, train_loader, epochs, device, lr)

    test_acc = evaluate(model, test_loader, device)
    print(f"[CENT-{model_name}] Final test accuracy = {test_acc:.4f}")

    return test_acc


# ===================== 9. 画图函数 =====================

def plot_summary_figure(fl_histories, fl_comm_bits, central_accs,
                        filename="summary_figure.png"):
    """
    生成一张 2x4 大图，总结：
      (a) Fed vs Fed-Comp 收敛曲线（MultiModalMLP）
      (b) Fed vs Fed-Comp 收敛曲线（EarlyFusionMLP）
      (c) Fed vs Fed-Comp 收敛曲线（MultiModalCNN1D）
      (d) 最终精度对比（Central / Fed / Fed-Comp）
      (e) 通信量对比（Fed / Fed-Comp, log-scale）
      (f) 通信压缩比（Fed / Fed-Comp）
      (g) 相对 Centralized 的精度差（Accuracy Gap）
      (h) Accuracy vs Communication 折中图

    fl_histories: dict[name -> list of acc per round]
    fl_comm_bits: dict[name -> total bits]
    central_accs: dict[base_model_name -> centralized acc]
    """
    base_models = ["MultiModalMLP", "EarlyFusionMLP", "MultiModalCNN1D"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # ---------- (a) 收敛曲线：MultiModalMLP ----------
    ax = axes[0, 0]
    mm_key = "MultiModalMLP"
    fed_key = f"Fed-{mm_key}"
    fed_comp_key = f"Fed-{mm_key}-Comp"

    if fed_key in fl_histories and fed_comp_key in fl_histories:
        rounds = range(1, len(fl_histories[fed_key]) + 1)
        ax.plot(rounds, fl_histories[fed_key],
                marker="o", linewidth=1.0, label="Fed-MLP")
        ax.plot(rounds, fl_histories[fed_comp_key],
                marker="s", linewidth=1.0, label="Fed-MLP-Comp")
    ax.set_xlabel("Global Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Convergence (MultiModalMLP)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # ---------- (b) 收敛曲线：EarlyFusionMLP ----------
    ax = axes[0, 1]
    base = "EarlyFusionMLP"
    fed_key = f"Fed-{base}"
    fed_comp_key = f"Fed-{base}-Comp"
    if fed_key in fl_histories and fed_comp_key in fl_histories:
        rounds = range(1, len(fl_histories[fed_key]) + 1)
        ax.plot(rounds, fl_histories[fed_key],
                marker="o", linewidth=1.0, label="Fed-Early")
        ax.plot(rounds, fl_histories[fed_comp_key],
                marker="s", linewidth=1.0, label="Fed-Early-Comp")
    ax.set_xlabel("Global Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(b) Convergence (EarlyFusionMLP)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # ---------- (c) 收敛曲线：MultiModalCNN1D ----------
    ax = axes[0, 2]
    base = "MultiModalCNN1D"
    fed_key = f"Fed-{base}"
    fed_comp_key = f"Fed-{base}-Comp"
    if fed_key in fl_histories and fed_comp_key in fl_histories:
        rounds = range(1, len(fl_histories[fed_key]) + 1)
        ax.plot(rounds, fl_histories[fed_key],
                marker="o", linewidth=1.0, label="Fed-CNN1D")
        ax.plot(rounds, fl_histories[fed_comp_key],
                marker="s", linewidth=1.0, label="Fed-CNN1D-Comp")
    ax.set_xlabel("Global Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(c) Convergence (MultiModalCNN1D)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # ---------- (d) 最终精度柱状图：Central vs Fed vs Fed-Comp ----------
    ax = axes[0, 3]
    labels = []
    acc_central = []
    acc_fed = []
    acc_fed_comp = []

    for base in base_models:
        key_c = base
        key_f = f"Fed-{base}"
        key_fc = f"Fed-{base}-Comp"

        if (key_c in central_accs and
            key_f in fl_histories and
            key_fc in fl_histories):
            labels.append(base)
            acc_central.append(central_accs[key_c])
            acc_fed.append(max(fl_histories[key_f]))
            acc_fed_comp.append(max(fl_histories[key_fc]))

    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, acc_central, width, label="Centralized")
    ax.bar(x,         acc_fed,     width, label="Fed")
    ax.bar(x + width, acc_fed_comp, width, label="Fed-Comp")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("(d) Final Accuracy Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    # ---------- (e) 通信量柱状图：Fed vs Fed-Comp (log scale) ----------
    ax = axes[1, 0]
    labels_comm = []
    comm_fed_mb = []
    comm_comp_mb = []

    for base in base_models:
        key_f = f"Fed-{base}"
        key_fc = f"Fed-{base}-Comp"
        if key_f in fl_comm_bits and key_fc in fl_comm_bits:
            labels_comm.append(base)
            b_f = fl_comm_bits[key_f] / 8 / 1024 / 1024  # bit -> MB
            b_fc = fl_comm_bits[key_fc] / 8 / 1024 / 1024
            comm_fed_mb.append(b_f)
            comm_comp_mb.append(b_fc)

    x2 = np.arange(len(labels_comm))
    width2 = 0.35

    bars_fed = ax.bar(x2 - width2 / 2, comm_fed_mb, width2, label="Fed")
    bars_comp = ax.bar(x2 + width2 / 2, comm_comp_mb, width2, label="Fed-Comp")

    ax.set_xticks(x2)
    ax.set_xticklabels(labels_comm, rotation=20)
    ax.set_ylabel("Total Communication (MB)")
    ax.set_title("(e) Communication Cost (Up + Down)")

    # 使用对数坐标，避免小柱子“看不见”
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)

    # 在柱子上标出大致数值，方便 reviewer 直接看
    for bar in list(bars_fed) + list(bars_comp):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.9,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.margins(y=0.2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    # ---------- (f) 相对通信效率增益：Fed-Comp / Fed ----------
    ax = axes[1, 1]

    labels_eff = []
    eff_ratio = []

    for base in base_models:
        key_f = f"Fed-{base}"
        key_fc = f"Fed-{base}-Comp"

        if (key_f in fl_histories and key_fc in fl_histories and
                key_f in fl_comm_bits and key_fc in fl_comm_bits):

            # best accuracy
            acc_f = max(fl_histories[key_f])
            acc_fc = max(fl_histories[key_fc])

            # communication MB
            comm_f = fl_comm_bits[key_f] / 8 / 1024 / 1024
            comm_fc = fl_comm_bits[key_fc] / 8 / 1024 / 1024

            eff_f = acc_f / comm_f if comm_f > 0 else 0.0
            eff_c = acc_fc / comm_fc if comm_fc > 0 else 0.0

            if eff_f > 0:
                labels_eff.append(base)
                eff_ratio.append(eff_c / eff_f)

    x3 = np.arange(len(labels_eff))
    bars = ax.bar(x3, eff_ratio)

    ax.set_xticks(x3)
    ax.set_xticklabels(labels_eff, rotation=20)
    ax.set_yscale("log")  # 可能是 10×~100×，开 log 看着更舒服
    ax.set_ylabel("Relative Efficiency (Fed-Comp / Fed)")
    ax.set_title("(f) Communication Efficiency Gain")

    # 在柱子上标出“多少倍”
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}×",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ---------- (g) 相对 Centralized 的精度差 ----------
    ax = axes[1, 2]
    labels_gap = []
    gap_fed = []
    gap_comp = []

    for base in base_models:
        key_c = base
        key_f = f"Fed-{base}"
        key_fc = f"Fed-{base}-Comp"
        if (key_c in central_accs and
            key_f in fl_histories and
            key_fc in fl_histories):
            acc_c = central_accs[key_c]
            acc_f = max(fl_histories[key_f])
            acc_fc = max(fl_histories[key_fc])

            labels_gap.append(base)
            gap_fed.append(acc_f - acc_c)
            gap_comp.append(acc_fc - acc_c)

    x4 = np.arange(len(labels_gap))
    width3 = 0.25

    ax.bar(x4 - width3/2, gap_fed, width3, label="Fed - Central")
    ax.bar(x4 + width3/2, gap_comp, width3, label="Fed-Comp - Central")

    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xticks(x4)
    ax.set_xticklabels(labels_gap, rotation=20)
    ax.set_ylabel("Accuracy Gap")
    ax.set_title("(g) Accuracy Gap w.r.t. Centralized")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    # ---------- (h) Accuracy vs Communication 散点图 ----------
    ax = axes[1, 3]
    comm_list = []
    acc_list = []

    for base in base_models:
        for suffix in ["", "-Comp"]:
            key = f"Fed-{base}{suffix}"
            if key in fl_histories and key in fl_comm_bits:
                comm_mb = fl_comm_bits[key] / 8 / 1024 / 1024
                acc_best = max(fl_histories[key])

                comm_list.append(comm_mb)
                acc_list.append(acc_best)

                ax.scatter(comm_mb, acc_best)
                ax.text(
                    comm_mb, acc_best,
                    key.replace("Fed-", ""),
                    fontsize=8, ha="left", va="bottom"
                )

    ax.set_xlabel("Total Communication (MB)")
    ax.set_ylabel("Best Test Accuracy")
    ax.set_title("(h) Accuracy vs Communication Trade-off")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 用 log x 轴，让左边挤在一起的点拉开
    if len(comm_list) > 0:
        ax.set_xscale("log")
        xmin = min(comm_list) * 0.7
        xmax = max(comm_list) * 1.3
        ax.set_xlim(xmin, xmax)

        # ======= 追加这三行 =======
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)  # 可选，防止内存里留太多 figure
    print(f"Saved summary figure to {filename}")


# ===================== 10. 主流程 =====================

def main():
    # -------- 修改成你自己电脑上的路径 --------
    RAMAN_FILE = "Raman.xlsx"     # 例如："/Users/lei/Desktop/Raman.xlsx"
    NMR_FILE = "NMR.xlsx"         # 例如："/Users/lei/Desktop/NMR.xlsx"

    # 你可以把文件复制成脚本同目录，然后直接写文件名
    assert os.path.exists(RAMAN_FILE), f"{RAMAN_FILE} not found"
    assert os.path.exists(NMR_FILE), f"{NMR_FILE} not found"

    X_raman, X_nmr, y, idx_to_label = load_raman_nmr_data(RAMAN_FILE, NMR_FILE)
    n_samples = X_raman.shape[0]
    n_classes = len(idx_to_label)

    print(f"Loaded Raman shape: {X_raman.shape}")
    print(f"Loaded NMR   shape: {X_nmr.shape}")
    print(f"Num samples: {n_samples}, Num classes: {n_classes}")
    print(f"Label mapping: {idx_to_label}")

    # 训练 / 测试划分（分层）
    train_idx, test_idx = split_train_test(
        n_samples, test_ratio=0.2, seed=42, y=y
    )

    # 客户端划分（联邦）
    clients_info = create_federated_clients(train_idx, n_clients=3, seed=0)
    for i, c in enumerate(clients_info):
        print(f"Client {i}: #samples={len(c['indices'])}, "
              f"has_raman={c['has_raman']}, has_nmr={c['has_nmr']}")

    # 构建联邦测试集（两个模态都可用）
    test_dataset_fl = HoneySpectraDataset(
        X_raman, X_nmr, y, indices=test_idx,
        has_raman=True, has_nmr=True
    )
    test_loader_fl = DataLoader(test_dataset_fl, batch_size=16, shuffle=False)

    # 设备（你可以改成 "mps" 使用苹果芯片 GPU）
    device = torch.device("cpu")
    # device = torch.device("mps")

    dim_r = X_raman.shape[1]
    dim_n = X_nmr.shape[1]

    # 将每个 client 包装成 DataLoader（联邦）
    client_loaders = []
    client_sizes = []
    for c in clients_info:
        ds = HoneySpectraDataset(
            X_raman, X_nmr, y,
            indices=c["indices"],
            has_raman=c["has_raman"],
            has_nmr=c["has_nmr"]
        )
        dl = DataLoader(ds, batch_size=16, shuffle=True)
        client_loaders.append((dl, c["has_raman"], c["has_nmr"]))
        client_sizes.append(len(c["indices"]))

    # 计算加权系数（按样本数）
    total_train_samples = sum(client_sizes)
    client_weights = [n / total_train_samples for n in client_sizes]

    # -------- 统一的模型构造列表 --------
    model_constructors = [
        ("MultiModalMLP",
         lambda: MultiModalMLP(dim_r, dim_n, n_classes)),
        ("EarlyFusionMLP",
         lambda: EarlyFusionMLP(dim_r, dim_n, n_classes)),
        ("MultiModalCNN1D",
         lambda: MultiModalCNN1D(dim_r, dim_n, n_classes)),
    ]

    # ===== 先跑集中式训练（Centralized）=====
    central_accs = {}  # name -> centralized accuracy
    for name, ctor in model_constructors:
        acc_c = run_centralized_experiment(
            model_name=name,
            model_ctor=ctor,
            X_raman=X_raman,
            X_nmr=X_nmr,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            device=device,
            epochs=100,
            batch_size=16,
            lr=1e-3,
        )
        central_accs[name] = acc_c

    # ===== 再跑联邦训练（Federated），含压缩 / 不压缩 =====
    num_rounds = 100
    local_epochs = 3
    lr = 1e-3

    fl_histories = {}   # name -> [acc per round]
    fl_comm_bits = {}   # name -> total bits

    for name, ctor in model_constructors:
        # 不压缩版本
        fed_name = f"Fed-{name}"
        hist = run_federated_experiment(
            model_name=fed_name,
            model_ctor=ctor,
            client_loaders=client_loaders,
            client_weights=client_weights,
            test_loader=test_loader_fl,
            device=device,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
            compress=False,
            comm_bits_dict=fl_comm_bits,
        )
        fl_histories[fed_name] = hist

        # 压缩版本（随机稀疏 + 随机量化）
        fed_comp_name = f"Fed-{name}-Comp"
        hist_c = run_federated_experiment(
            model_name=fed_comp_name,
            model_ctor=ctor,
            client_loaders=client_loaders,
            client_weights=client_weights,
            test_loader=test_loader_fl,
            device=device,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
            compress=True,
            sparsity=0.1,   # 保留 10% 参数
            num_bits=8,     # 8-bit 量化
            comm_bits_dict=fl_comm_bits,
        )
        fl_histories[fed_comp_name] = hist_c

    print("\n===== All Centralized & Federated Experiments Finished =====")

    # -------- 集中式 RandomForest 对比（可选传统 baseline） --------
    if HAVE_SKLEARN:
        run_random_forest_baseline(X_raman, X_nmr, y, train_idx, test_idx)

    # -------- 生成总图：收敛 + 精度 + 通信 + 折中 --------
    plot_summary_figure(
        fl_histories=fl_histories,
        fl_comm_bits=fl_comm_bits,
        central_accs=central_accs,
        filename="summary_figure.png"
    )


if __name__ == "__main__":
    main()
