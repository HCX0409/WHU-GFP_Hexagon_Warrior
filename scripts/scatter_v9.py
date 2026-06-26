"""
V9 LoRA 72k 广谱散点图
加载ESM-2 + LoRA适配器做推理, 亮度区间均匀采样
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

from src.config import LOCAL_ESM2_DIR, MODELS_DIR, PROCESSED_DIR, get_device

# 常量
GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {t: i for i, t in enumerate(GFP_TYPE_LIST)}
WT_BRIGHTNESS = {'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258}

MODEL_DIR = MODELS_DIR / "ESM2_LoRA_V9_72k_HCX_260618"
DATA_CSV = PROCESSED_DIR / "v7_72k_quality_data.csv"

COLORS = {'avGFP': '#2196F3', 'amacGFP': '#4CAF50', 'cgreGFP': '#FF9800', 'ppluGFP': '#9C27B0'}


class SeqDataset(Dataset):
    def __init__(self, seqs, tokenizer, max_len=250):
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
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class ESM2LoRARegressorV9(nn.Module):
    """V9模型: LoRA + LayerNorm + 深层MLP头"""
    def __init__(self, base_model, num_gfp_types=4):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.pool_norm = nn.LayerNorm(hidden_size)
        input_dim = hidden_size + num_gfp_types
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.num_gfp_types = num_gfp_types

    def forward(self, input_ids, attention_mask, gfp_type_idx):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.pool_norm(pooled)
        type_onehot = torch.zeros(pooled.size(0), self.num_gfp_types, device=pooled.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        combined = torch.cat([pooled, type_onehot], dim=1)
        return self.regressor(combined).squeeze(-1)


def uniform_sample_indices(true_brightness, n_target=3000, n_bins=20):
    """按亮度均匀分层采样"""
    bins = np.linspace(true_brightness.min(), true_brightness.max(), n_bins + 1)
    indices = []
    per_bin = n_target // n_bins
    for i in range(n_bins):
        mask = (true_brightness >= bins[i]) & (true_brightness < bins[i+1])
        if i == n_bins - 1:
            mask = (true_brightness >= bins[i]) & (true_brightness <= bins[i+1])
        bin_idx = np.where(mask)[0]
        if len(bin_idx) <= per_bin:
            indices.extend(bin_idx)
        else:
            indices.extend(np.random.choice(bin_idx, per_bin, replace=False))
    return np.array(indices)


def main():
    device = get_device()
    print(f"Device: {device}")

    # 1. 加载数据
    print("加载数据...")
    df = pd.read_csv(DATA_CSV)
    true_brightness = df['brightness'].values
    gfp_types = df['gfp_type'].values
    type_indices = np.array([GFP_TYPE_TO_IDX[gt] for gt in gfp_types], dtype=np.int64)
    seqs = df['sequence'].tolist()

    # 2. 加载V9 LoRA模型
    print("加载ESM-2 + LoRA...")
    model_name = str(LOCAL_ESM2_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # 加载LoRA适配器
    peft_model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
    peft_model = peft_model.merge_and_unload()  # 合并LoRA权重到base model

    model = ESM2LoRARegressorV9(peft_model, num_gfp_types=4)
    regressor_state = torch.load(MODEL_DIR / "regressor.pt", map_location='cpu', weights_only=True)
    # 加载regressor和pool_norm权重
    full_state = {}
    for k, v in regressor_state.items():
        full_state[f'regressor.{k}'] = v
    # 尝试加载pool_norm
    model_state = model.state_dict()
    for k in ['pool_norm.weight', 'pool_norm.bias']:
        if k in model_state:
            full_state[k] = model_state[k]
    model.load_state_dict(full_state, strict=False)
    model = model.to(device)
    model.eval()
    print("  模型加载完成 (LoRA已merge)")

    # 3. 推理
    print("推理中 (72k序列)...")
    ds = SeqDataset(seqs, tokenizer, max_len=250)
    loader = DataLoader(ds, batch_size=48, num_workers=4, pin_memory=True)

    type_tensor = torch.tensor(type_indices, dtype=torch.long)
    all_pred_delta = []

    batch_start = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="V9推理"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bs = input_ids.size(0)
            gfp_type = type_tensor[batch_start:batch_start + bs].to(device)
            batch_start += bs

            with torch.amp.autocast('cuda'):
                pred = model(input_ids, attention_mask, gfp_type)
            all_pred_delta.extend(pred.cpu().float().numpy())

    pred_delta = np.array(all_pred_delta)
    wt_brightness = np.array([WT_BRIGHTNESS[gt] for gt in gfp_types])
    pred_brightness = pred_delta + wt_brightness

    # 4. 全局指标
    print(f"\n{'='*60}")
    r2_all = r2_score(true_brightness, pred_brightness)
    r_all = pearsonr(true_brightness, pred_brightness)[0]
    mae_all = mean_absolute_error(true_brightness, pred_brightness)
    print(f"全局 (N={len(df)}): R²={r2_all:.4f}, r={r_all:.4f}, MAE={mae_all:.4f}")

    # 5. 均匀采样散点图
    print("\n生成广谱散点图 (均匀采样)...")
    np.random.seed(42)
    sample_idx = uniform_sample_indices(true_brightness, n_target=3000, n_bins=20)

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    lims = [1.0, 5.0]

    # --- 图1: 全局 ---
    ax = fig.add_subplot(gs[0, :2])
    si = sample_idx
    for gt in GFP_TYPE_LIST:
        mask = gfp_types[si] == gt
        ax.scatter(true_brightness[si][mask], pred_brightness[si][mask],
                   c=COLORS[gt], alpha=0.4, s=12, label=gt)
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5, label='y=x')
    ax.set_xlabel('True Brightness (log10)', fontsize=12)
    ax.set_ylabel('Predicted Brightness', fontsize=12)
    ax.set_title(f'V9 LoRA - All Types (均匀采样 N={len(si)})\n'
                 f'R²={r2_all:.3f}, r={r_all:.3f}, MAE={mae_all:.3f}', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(True, alpha=0.2)

    # --- 图2: 统计面板 ---
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    stats_text = "V9 LoRA 分类型指标\n" + "─" * 30
    for gt in GFP_TYPE_LIST:
        mask = gfp_types == gt
        r2 = r2_score(true_brightness[mask], pred_brightness[mask])
        r = pearsonr(true_brightness[mask], pred_brightness[mask])[0]
        mae = mean_absolute_error(true_brightness[mask], pred_brightness[mask])
        stats_text += f"\n\n{gt} (N={mask.sum()})"
        stats_text += f"\n  R² = {r2:.3f}"
        stats_text += f"\n  r  = {r:.3f}"
        stats_text += f"\n  MAE= {mae:.3f}"
    hi_mask = true_brightness >= 4.0
    if hi_mask.sum() > 10:
        r2_hi = r2_score(true_brightness[hi_mask], pred_brightness[hi_mask])
        r_hi = pearsonr(true_brightness[hi_mask], pred_brightness[hi_mask])[0]
        stats_text += f"\n\n高亮度(≥4.0) N={hi_mask.sum()}"
        stats_text += f"\n  R² = {r2_hi:.3f}"
        stats_text += f"\n  r  = {r_hi:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- 图3-6: 各类型独立散点图 ---
    for i, gt in enumerate(GFP_TYPE_LIST):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        mask_all = gfp_types == gt
        true_gt = true_brightness[mask_all]
        pred_gt = pred_brightness[mask_all]
        si_gt = uniform_sample_indices(true_gt, n_target=800, n_bins=15)
        ax.scatter(true_gt[si_gt], pred_gt[si_gt], c=COLORS[gt], alpha=0.4, s=15)
        ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5)
        r2 = r2_score(true_gt, pred_gt)
        r = pearsonr(true_gt, pred_gt)[0]
        mae = mean_absolute_error(true_gt, pred_gt)
        ax.set_xlabel('True Brightness', fontsize=11)
        ax.set_ylabel('Predicted Brightness', fontsize=11)
        ax.set_title(f'{gt} (N_sample={len(si_gt)})\n'
                     f'R²={r2:.3f}, r={r:.3f}, MAE={mae:.3f}', fontsize=12)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.grid(True, alpha=0.2)

    # --- 图7: 高亮度区 ---
    ax = fig.add_subplot(gs[2, 2])
    hi_cg = (gfp_types == 'cgreGFP') & (true_brightness >= 4.0)
    hi_pp = (gfp_types == 'ppluGFP') & (true_brightness >= 4.0)
    if hi_cg.sum() > 0:
        ax.scatter(true_brightness[hi_cg], pred_brightness[hi_cg],
                   c=COLORS['cgreGFP'], alpha=0.3, s=15, label=f'cgreGFP (N={hi_cg.sum()})')
    if hi_pp.sum() > 0:
        ax.scatter(true_brightness[hi_pp], pred_brightness[hi_pp],
                   c=COLORS['ppluGFP'], alpha=0.3, s=15, label=f'ppluGFP (N={hi_pp.sum()})')
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5)
    hi_all = true_brightness >= 4.0
    if hi_all.sum() > 10:
        r2_hi = r2_score(true_brightness[hi_all], pred_brightness[hi_all])
        r_hi = pearsonr(true_brightness[hi_all], pred_brightness[hi_all])[0]
        ax.set_title(f'高亮度区 (≥4.0, N={hi_all.sum()})\nR²={r2_hi:.3f}, r={r_hi:.3f}', fontsize=12)
    ax.set_xlabel('True Brightness', fontsize=11)
    ax.set_ylabel('Predicted Brightness', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(True, alpha=0.2)

    out_path = Path(__file__).resolve().parent.parent / "assets" / "v9_scatter_uniform.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n散点图已保存: {out_path}")


if __name__ == "__main__":
    main()
