"""
Phase 1.6-Frozen: ESM-2冻结特征提取 + 独立深度学习预测亮度
不用LoRA, 完全避开gradient checkpointing问题

流程:
  1. 用冻结的ESM-2提取所有序列的1280维mean-pooling嵌入 (一次性, 缓存)
  2. 用嵌入+GFP类型one-hot训练一个深层MLP预测delta
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
import joblib

from src.config import (
    PROCESSED_DIR, MODELS_DIR, LOCAL_ESM2_DIR,
    RANDOM_SEED, AUTHOR, DATE_STR, seed_everything, get_device
)

# ======================== 常量 ========================
GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {t: i for i, t in enumerate(GFP_TYPE_LIST)}

WT_BRIGHTNESS = {
    'avGFP': 3.7192,
    'amacGFP': 3.9707,
    'cgreGFP': 4.4969,
    'ppluGFP': 4.2258
}

EMBEDDING_CACHE = PROCESSED_DIR / "v7_esm2_embeddings_72k.joblib"


class SeqDataset(Dataset):
    """序列→ESM-2输入的Dataset (模块级, 支持Windows pickle)"""
    def __init__(self, seqs, tokenizer, max_len):
        self.seqs = seqs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.seqs[idx], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


def concordance_correlation_coefficient(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / (denominator + 1e-8)


# ======================== Step 1: ESM-2特征提取 ========================
@torch.no_grad()
def extract_esm2_embeddings(seqs, tokenizer, model, device, batch_size=64, max_len=250):
    """用冻结的ESM-2提取所有序列的mean-pooling嵌入 (1280维)"""
    model.eval()
    all_embeddings = []

    loader = DataLoader(
        SeqDataset(seqs, tokenizer, max_len),
        batch_size=batch_size, num_workers=8, pin_memory=True
    )

    for batch in tqdm(loader, desc="提取ESM-2嵌入"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, 1280)

        # mean pooling (只对有意义的token)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, 1280)

        all_embeddings.append(pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)  # (N, 1280)


# ======================== Step 2: V7.5 残差预测MLP ========================
class ResidualBrightnessMLP(nn.Module):
    """
    V7.5: 简洁架构 + 残差预测 + 可学习类型基线
    - V7的简洁concat(one-hot) + V8的LayerNorm
    - 残差预测: 粗delta + 残差修正 = 精调delta
    - 绝对亮度 = delta_final + 可学习类型基线
    """

    def __init__(self, input_dim=1280, num_gfp_types=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_gfp_types = num_gfp_types

        # 输入归一化 (V8有效改进)
        self.input_norm = nn.LayerNorm(input_dim)

        # V7式concat输入
        total_input = input_dim + num_gfp_types

        # 主干网络
        self.backbone = nn.Sequential(
            nn.Linear(total_input, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
        )

        # 粗预测头: 主干 → 粗粒度delta
        self.coarse_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # 残差修正头: 主干特征 → 小修正值 (初始化小权重)
        self.residual_head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1)
        )
        # 残差头初始化小权重, 开始接近零输出
        for m in self.residual_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

        # 可学习类型基线偏移 (从WT亮度初始化)
        self.type_baseline = nn.Parameter(torch.tensor([3.7192, 3.9707, 4.4969, 4.2258]))

    def forward(self, embeddings, gfp_type_idx):
        """
        返回: (coarse_delta, delta_final, pred_abs_brightness)
        """
        B = embeddings.size(0)
        x = self.input_norm(embeddings)

        # concat one-hot (V7式)
        type_onehot = torch.zeros(B, self.num_gfp_types, device=embeddings.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        x = torch.cat([x, type_onehot], dim=1)

        # 主干
        features = self.backbone(x)

        # 粗预测 + 残差修正
        coarse_delta = self.coarse_head(features).squeeze(-1)
        residual = self.residual_head(features).squeeze(-1)
        delta_final = coarse_delta + residual

        # 绝对亮度 = delta_final + 可学习类型基线
        baseline = self.type_baseline[gfp_type_idx]
        pred_abs = delta_final + baseline

        return coarse_delta, delta_final, pred_abs


# ======================== 训练循环 ========================
def train_mlp(X_train_emb, Y_train, T_train, Abs_train,
              X_val_emb, Y_val, T_val, Abs_val,
              X_test_emb, Y_test, T_test, Abs_test,
              device, epochs=100, batch_size=256, lr=1e-3,
              aux_weight=0.3):
    """
    训练V7.5残差预测MLP
    Y_*: delta标签, Abs_*: 绝对亮度标签
    """

    # 过采样: delta>0.5 → 2x, delta<-0.5 → 1.5x
    high_mask = Y_train > 0.5
    low_mask = Y_train < -0.5
    mid_mask = ~(high_mask | low_mask)

    X_high, Y_high, T_high, A_high = X_train_emb[high_mask], Y_train[high_mask], T_train[high_mask], Abs_train[high_mask]
    X_low, Y_low, T_low, A_low = X_train_emb[low_mask], Y_train[low_mask], T_train[low_mask], Abs_train[low_mask]
    X_mid, Y_mid, T_mid, A_mid = X_train_emb[mid_mask], Y_train[mid_mask], T_train[mid_mask], Abs_train[mid_mask]

    # 拼接: high 2x, low 1.5x, mid 1x
    X_over = np.concatenate([X_mid, X_high, X_high, X_low, X_low[:len(X_low)//2]], axis=0)
    Y_over = np.concatenate([Y_mid, Y_high, Y_high, Y_low, Y_low[:len(Y_low)//2]], axis=0)
    T_over = np.concatenate([T_mid, T_high, T_high, T_low, T_low[:len(T_low)//2]], axis=0)
    A_over = np.concatenate([A_mid, A_high, A_high, A_low, A_low[:len(A_low)//2]], axis=0)

    print(f"  过采样后训练集: {len(X_over)} 条")
    print(f"    delta>0.5: {high_mask.sum()} → {high_mask.sum()*2}")
    print(f"    delta<-0.5: {low_mask.sum()} → {low_mask.sum() + low_mask.sum()//2}")
    print(f"    中间: {mid_mask.sum()}")

    # 加权: delta>0.5→6x, 中间→2x, delta<-0.5→2.5x (V7验证有效的权重)
    weights = np.ones(len(Y_over), dtype=np.float32)
    weights[Y_over > 0.5] *= 6.0
    weights[Y_over < -0.5] *= 2.5
    mid_over_mask = (Y_over >= -0.5) & (Y_over <= 0.5)
    weights[mid_over_mask] *= 2.0

    # DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_over, dtype=torch.float32),
        torch.tensor(Y_over, dtype=torch.float32),
        torch.tensor(T_over, dtype=torch.long),
        torch.tensor(A_over, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_emb, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
        torch.tensor(T_val, dtype=torch.long),
        torch.tensor(Abs_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=4, pin_memory=True, persistent_workers=True)

    # 模型
    model = ResidualBrightnessMLP(input_dim=1280, num_gfp_types=4).to(device)
    delta_criterion = nn.HuberLoss(delta=0.5, reduction='none')
    abs_criterion = nn.MSELoss(reduction='none')  # 绝对亮度用MSE
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_r2 = -999
    best_model_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        # --- 训练 ---
        model.train()
        train_losses = []
        for emb, labels_delta, gfp_type, labels_abs, w in train_loader:
            emb = emb.to(device)
            labels_delta = labels_delta.to(device)
            gfp_type = gfp_type.to(device)
            labels_abs = labels_abs.to(device)
            w = w.to(device)

            optimizer.zero_grad()
            coarse_delta, delta_final, pred_abs = model(emb, gfp_type)

            # 联合损失: delta_final + 绝对亮度
            loss_delta = (delta_criterion(delta_final, labels_delta) * w).mean()
            loss_abs = (abs_criterion(pred_abs, labels_abs) * w).mean()
            loss = loss_delta + aux_weight * loss_abs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_preds_delta, val_preds_abs, val_labels_delta, val_labels_abs = [], [], [], []
        with torch.no_grad():
            for emb, labels_delta, gfp_type, labels_abs in val_loader:
                emb, gfp_type = emb.to(device), gfp_type.to(device)
                _, delta_final, pred_abs = model(emb, gfp_type)
                val_preds_delta.extend(delta_final.cpu().numpy())
                val_preds_abs.extend(pred_abs.cpu().numpy())
                val_labels_delta.extend(labels_delta.numpy())
                val_labels_abs.extend(labels_abs.numpy())

        val_preds_delta = np.array(val_preds_delta)
        val_preds_abs = np.array(val_preds_abs)
        val_labels_delta = np.array(val_labels_delta)
        val_labels_abs = np.array(val_labels_abs)

        # 主指标: delta R² + 绝对亮度 R²
        r2_delta = r2_score(val_labels_delta, val_preds_delta)
        r2_abs = r2_score(val_labels_abs, val_preds_abs)
        r_delta = pearsonr(val_labels_delta, val_preds_delta)[0]
        r_abs = pearsonr(val_labels_abs, val_preds_abs)[0]

        # 综合评分: 0.5*delta_r2 + 0.5*abs_r2
        combined_r2 = 0.5 * r2_delta + 0.5 * r2_abs

        # 分类型评估 (绝对亮度)
        type_stats = []
        for gt in GFP_TYPE_LIST:
            mask = np.array([GFP_TYPE_TO_IDX[gt] == t for t in T_val])
            if mask.sum() > 5:
                tr = r2_score(val_labels_abs[mask], val_preds_abs[mask])
                tr_p = pearsonr(val_labels_abs[mask], val_preds_abs[mask])[0]
                type_stats.append(f"  {gt}: R²_abs={tr:.3f}, r={tr_p:.3f} (N={mask.sum()})")

        marker = ""
        if combined_r2 > best_r2:
            best_r2 = combined_r2
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " ✔ 新最佳"
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {np.mean(train_losses):.4f} | "
                  f"R²_delta: {r2_delta:.4f} | R²_abs: {r2_abs:.4f} | "
                  f"r_delta: {r_delta:.4f} | r_abs: {r_abs:.4f} | 综合: {combined_r2:.4f}{marker}")
            if (epoch + 1) % 10 == 0 or marker:
                for s in type_stats:
                    print(f"    {s}")

        if no_improve >= patience:
            print(f"  ⏸ 早停: {patience} epoch无改善 (最佳综合R²={best_r2:.4f})")
            break

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    # === 测试集评估 ===
    test_ds = TensorDataset(
        torch.tensor(X_test_emb, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
        torch.tensor(T_test, dtype=torch.long),
        torch.tensor(Abs_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, num_workers=4, persistent_workers=True)
    test_preds_delta, test_preds_abs = [], []
    test_labels_delta, test_labels_abs = [], []
    with torch.no_grad():
        for emb, labels_delta, gfp_type, labels_abs in test_loader:
            emb, gfp_type = emb.to(device), gfp_type.to(device)
            _, delta_final, pred_abs = model(emb, gfp_type)
            test_preds_delta.extend(delta_final.cpu().numpy())
            test_preds_abs.extend(pred_abs.cpu().numpy())
            test_labels_delta.extend(labels_delta.numpy())
            test_labels_abs.extend(labels_abs.numpy())

    test_preds_delta = np.array(test_preds_delta)
    test_preds_abs = np.array(test_preds_abs)
    test_labels_delta = np.array(test_labels_delta)
    test_labels_abs = np.array(test_labels_abs)

    print(f"\n{'='*60}")
    print(f"📊 测试集 (delta_final=粗+残差): R²={r2_score(test_labels_delta, test_preds_delta):.4f}, "
          f"r={pearsonr(test_labels_delta, test_preds_delta)[0]:.4f}, "
          f"MAE={mean_absolute_error(test_labels_delta, test_preds_delta):.4f}")
    print(f"📊 测试集 (绝对亮度=delta+基线): R²={r2_score(test_labels_abs, test_preds_abs):.4f}, "
          f"r={pearsonr(test_labels_abs, test_preds_abs)[0]:.4f}, "
          f"MAE={mean_absolute_error(test_labels_abs, test_preds_abs):.4f}")

    # 对比: 固定WT基线方式
    test_abs_fixed_wt = np.array([d + WT_BRIGHTNESS[GFP_TYPE_LIST[t]] for d, t in zip(test_preds_delta, T_test)])
    print(f"📊 测试集 (delta+固定WT): R²={r2_score(test_labels_abs, test_abs_fixed_wt):.4f}, "
          f"r={pearsonr(test_labels_abs, test_abs_fixed_wt)[0]:.4f}")

    # 显示可学习基线 vs 固定WT
    learned_baseline = model.type_baseline.detach().cpu().numpy()
    print(f"\n  可学习基线: {dict(zip(GFP_TYPE_LIST, [f'{b:.4f}' for b in learned_baseline]))}")
    print(f"  固定WT基线: {dict(zip(GFP_TYPE_LIST, [f'{b:.4f}' for b in [3.7192, 3.9707, 4.4969, 4.2258]]))}")

    for gt in GFP_TYPE_LIST:
        mask = np.array([GFP_TYPE_TO_IDX[gt] == t for t in T_test])
        if mask.sum() > 5:
            tr_d = r2_score(test_labels_delta[mask], test_preds_delta[mask])
            tr_a = r2_score(test_labels_abs[mask], test_preds_abs[mask])
            tr_p = pearsonr(test_labels_abs[mask], test_preds_abs[mask])[0]
            print(f"  {gt}: R²_delta={tr_d:.4f}, R²_abs={tr_a:.4f}, r_abs={tr_p:.4f} (N={mask.sum()})")

    # 高亮度区间 (brightness>=4.0)
    hi_mask = test_labels_abs >= 4.0
    if hi_mask.sum() > 10:
        hi_r2_abs = r2_score(test_labels_abs[hi_mask], test_preds_abs[hi_mask])
        hi_r_abs = pearsonr(test_labels_abs[hi_mask], test_preds_abs[hi_mask])[0]
        hi_r2_fixed = r2_score(test_labels_abs[hi_mask], test_abs_fixed_wt[hi_mask])
        print(f"  高亮度(>=4.0): R²_abs(可学习基线)={hi_r2_abs:.4f}, r={hi_r_abs:.4f}, "
              f"R²(固定WT)={hi_r2_fixed:.4f} (N={hi_mask.sum()})")

    return model


# ======================== 主函数 ========================
def main():
    seed_everything(RANDOM_SEED)
    import os
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    device = get_device()
    print(f"Device: {device}, CPU线程(intra): {torch.get_num_threads()}, interop: {torch.get_num_interop_threads()}")

    # 1. 加载数据
    print("=" * 60)
    print("Step 1: 加载训练数据")
    
    # 优先使用72k数据, 否则回退到v6数据
    csv_72k = PROCESSED_DIR / "v7_72k_quality_data.csv"
    csv_v6 = PROCESSED_DIR / "v6_training_data.csv"
    
    if csv_72k.exists():
        csv_path = csv_72k
        print(f"  使用72k数据: {csv_path}")
    elif csv_v6.exists():
        csv_path = csv_v6
        print(f"  使用V6数据: {csv_path}")
    else:
        raise FileNotFoundError(f"找不到训练数据: {csv_72k} 或 {csv_v6}")
    
    df = pd.read_csv(csv_path)
    seqs = df['sequence'].tolist()
    gfp_types = df['gfp_type'].tolist()
    brightness = df['brightness'].values
    labels = np.array([b - WT_BRIGHTNESS[gt] for b, gt in zip(brightness, gfp_types)], dtype=np.float32)
    type_indices = np.array([GFP_TYPE_TO_IDX[gt] for gt in gfp_types], dtype=np.int64)

    print(f"  总样本数: {len(df)}")
    print(f"  Delta统计: mean={labels.mean():.3f}, std={labels.std():.3f}, "
          f"min={labels.min():.3f}, max={labels.max():.3f}")

    # 2. 提取/加载ESM-2嵌入
    print("=" * 60)
    print("Step 2: 提取ESM-2嵌入 (1280维)")

    if EMBEDDING_CACHE.exists():
        print(f"  找到缓存: {EMBEDDING_CACHE}")
        cache = joblib.load(EMBEDDING_CACHE)
        embeddings = cache['embeddings']
        cached_seqs = cache['sequences']
        # 验证缓存是否匹配
        if len(cached_seqs) == len(seqs) and cached_seqs[0] == seqs[0] and cached_seqs[-1] == seqs[-1]:
            print(f"  ✅ 缓存匹配, 直接加载 ({embeddings.shape})")
        else:
            print("  ⚠️ 缓存不匹配, 重新提取")
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print("  加载ESM-2模型...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_ESM2_DIR)
        esm2_model = AutoModel.from_pretrained(LOCAL_ESM2_DIR)
        esm2_model.to(device)
        esm2_model.eval()

        embeddings = extract_esm2_embeddings(seqs, tokenizer, esm2_model, device, batch_size=48)
        print(f"  嵌入形状: {embeddings.shape}")

        # 缓存
        joblib.dump({'embeddings': embeddings, 'sequences': seqs}, EMBEDDING_CACHE)
        print(f"  💾 缓存已保存: {EMBEDDING_CACHE}")

        # 释放ESM-2显存
        del esm2_model, tokenizer
        torch.cuda.empty_cache()

    # 3. 划分训练/验证/测试集
    print("=" * 60)
    print("Step 3: 划分数据集")

    abs_brightness = brightness.astype(np.float32)  # 绝对亮度标签

    indices = np.arange(len(embeddings))
    idx_train, idx_temp = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=RANDOM_SEED)

    X_train_emb, Y_train, T_train = embeddings[idx_train], labels[idx_train], type_indices[idx_train]
    X_val_emb, Y_val, T_val = embeddings[idx_val], labels[idx_val], type_indices[idx_val]
    X_test_emb, Y_test, T_test = embeddings[idx_test], labels[idx_test], type_indices[idx_test]
    Abs_train = abs_brightness[idx_train]
    Abs_val = abs_brightness[idx_val]
    Abs_test = abs_brightness[idx_test]

    print(f"  训练: {len(X_train_emb)}, 验证: {len(X_val_emb)}, 测试: {len(X_test_emb)}")

    # 4. 训练V7.5残差预测MLP
    print("=" * 60)
    print("Step 4: 训练V7.5残差预测MLP")
    print(f"  架构: ResidualBrightnessMLP (粗delta + 残差修正 + 可学习基线)")
    print(f"  加权: delta>0.5→6x, 中间→2x, delta<-0.5→2.5x")
    print(f"  绝对亮度损失权重: {0.3}")

    model = train_mlp(
        X_train_emb, Y_train, T_train, Abs_train,
        X_val_emb, Y_val, T_val, Abs_val,
        X_test_emb, Y_test, T_test, Abs_test,
        device, epochs=100, batch_size=256, lr=1e-3,
        aux_weight=0.3
    )

    # 5. 保存模型
    print("=" * 60)
    print("Step 5: 保存模型")
    save_dir = MODELS_DIR / f"ESM2_ResidualMLP_V75_72k_{AUTHOR}_{DATE_STR}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "mlp_model.pt")
    joblib.dump({
        'WT_BRIGHTNESS': WT_BRIGHTNESS,
        'GFP_TYPE_LIST': GFP_TYPE_LIST,
        'GFP_TYPE_TO_IDX': GFP_TYPE_TO_IDX
    }, save_dir / "model_config.joblib")

    print(f"  💾 ResidualBrightnessMLP已保存: {save_dir}")
    print(f"  💡 推理: ESM-2嵌入 → LayerNorm+concat → 残差delta + 可学习基线")
    print(f"\n{'='*60}")
    print(f"完成!")


if __name__ == "__main__":
    main()
