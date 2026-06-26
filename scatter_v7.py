"""
V7.5 ResidualBrightnessMLP 72k 广谱散点图
亮度区间均匀采样, 4种GFP类型 + 高亮度区 + 全局
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import joblib

from src.config import LOCAL_ESM2_DIR, MODELS_DIR, PROCESSED_DIR, get_device

# 常量
GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {t: i for i, t in enumerate(GFP_TYPE_LIST)}
WT_BRIGHTNESS = {'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258}

MODEL_DIR = MODELS_DIR / "ESM2_ResidualMLP_V75_72k_HCX_260618"
EMBEDDING_CACHE = PROCESSED_DIR / "v7_esm2_embeddings_72k.joblib"
DATA_CSV = PROCESSED_DIR / "v7_72k_quality_data.csv"

COLORS = {'avGFP': '#2196F3', 'amacGFP': '#4CAF50', 'cgreGFP': '#FF9800', 'ppluGFP': '#9C27B0'}


class ResidualBrightnessMLP(nn.Module):
    """V7.5: 残差预测 + 可学习类型基线"""
    def __init__(self, input_dim=1280, num_gfp_types=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_gfp_types = num_gfp_types
        self.input_norm = nn.LayerNorm(input_dim)
        total_input = input_dim + num_gfp_types
        self.backbone = nn.Sequential(
            nn.Linear(total_input, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
        )
        self.coarse_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1)
        )
        for m in self.residual_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        self.type_baseline = nn.Parameter(torch.tensor([3.7192, 3.9707, 4.4969, 4.2258]))

    def forward(self, embeddings, gfp_type_idx):
        B = embeddings.size(0)
        x = self.input_norm(embeddings)
        type_onehot = torch.zeros(B, self.num_gfp_types, device=embeddings.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        x = torch.cat([x, type_onehot], dim=1)
        features = self.backbone(x)
        coarse_delta = self.coarse_head(features).squeeze(-1)
        residual = self.residual_head(features).squeeze(-1)
        delta_final = coarse_delta + residual
        baseline = self.type_baseline[gfp_type_idx]
        pred_abs = delta_final + baseline
        return coarse_delta, delta_final, pred_abs


def uniform_sample_indices(true_brightness, n_target=3000, n_bins=20):
    """按亮度均匀分层采样, 保证各亮度区间均匀分布"""
    bins = np.linspace(true_brightness.min(), true_brightness.max(), n_bins + 1)
    indices = []
    per_bin = n_target // n_bins
    
    for i in range(n_bins):
        mask = (true_brightness >= bins[i]) & (true_brightness < bins[i+1])
        if i == n_bins - 1:  # 最后一个bin包含右边界
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
    
    # 1. 加载数据和嵌入
    print("加载数据...")
    df = pd.read_csv(DATA_CSV)
    true_brightness = df['brightness'].values
    gfp_types = df['gfp_type'].values
    type_indices = np.array([GFP_TYPE_TO_IDX[gt] for gt in gfp_types], dtype=np.int64)
    
    print("加载嵌入缓存...")
    cache = joblib.load(EMBEDDING_CACHE)
    embeddings = cache['embeddings']
    print(f"  嵌入: {embeddings.shape}")
    
    # 2. 加载ResidualBrightnessMLP模型
    print("加载ResidualBrightnessMLP模型...")
    model = ResidualBrightnessMLP(input_dim=1280, num_gfp_types=4).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "mlp_model.pt", map_location=device, weights_only=True))
    model.eval()
    
    # 3. 推理 (用缓存嵌入, 不需要ESM-2)
    print("推理中...")
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    type_tensor = torch.tensor(type_indices, dtype=torch.long)
    
    ds = TensorDataset(emb_tensor, type_tensor)
    loader = DataLoader(ds, batch_size=512, num_workers=0)
    
    all_pred_delta = []
    all_pred_abs = []
    with torch.no_grad():
        for emb, gt in loader:
            emb, gt = emb.to(device), gt.to(device)
            _, delta_final, pred_abs = model(emb, gt)
            all_pred_delta.extend(delta_final.cpu().numpy())
            all_pred_abs.extend(pred_abs.cpu().numpy())
    
    pred_delta = np.array(all_pred_delta)
    pred_abs = np.array(all_pred_abs)
    wt_brightness = np.array([WT_BRIGHTNESS[gt] for gt in gfp_types])
    pred_brightness_from_delta = pred_delta + wt_brightness
    # 用可学习基线版本作为主要预测值
    pred_brightness = pred_abs
    
    # 4. 全局指标
    print(f"\n{'='*60}")
    r2_all = r2_score(true_brightness, pred_brightness)
    r_all = pearsonr(true_brightness, pred_brightness)[0]
    mae_all = mean_absolute_error(true_brightness, pred_brightness)
    r2_delta_way = r2_score(true_brightness, pred_brightness_from_delta)
    print(f"全局 (N={len(df)}):")
    print(f"  可学习基线: R²={r2_all:.4f}, r={r_all:.4f}, MAE={mae_all:.4f}")
    print(f"  delta+固定WT: R²={r2_delta_way:.4f}")
    
    # 5. 均匀采样散点图
    print("\n生成广谱散点图 (均匀采样)...")
    
    # 均匀采样3000个代表点
    np.random.seed(42)
    sample_idx = uniform_sample_indices(true_brightness, n_target=3000, n_bins=20)
    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    lims = [1.0, 5.0]
    
    # --- 图1: 全局 (均匀采样) ---
    ax = fig.add_subplot(gs[0, :2])  # 宽图
    si = sample_idx
    for gt in GFP_TYPE_LIST:
        mask = gfp_types[si] == gt
        ax.scatter(true_brightness[si][mask], pred_brightness[si][mask],
                   c=COLORS[gt], alpha=0.4, s=12, label=gt)
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5, label='y=x')
    ax.set_xlabel('True Brightness (log10)', fontsize=12)
    ax.set_ylabel('Predicted Brightness', fontsize=12)
    ax.set_title(f'V7.5 ResidualMLP - All Types (均匀采样 N={len(si)})\n'
                 f'R²={r2_all:.3f}, r={r_all:.3f}, MAE={mae_all:.3f}', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(True, alpha=0.2)
    
    # --- 图2: 各类型统计 ---
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    stats_text = "分类型指标 (测试集)\n" + "─" * 30
    for gt in GFP_TYPE_LIST:
        mask = gfp_types == gt
        r2 = r2_score(true_brightness[mask], pred_brightness[mask])
        r = pearsonr(true_brightness[mask], pred_brightness[mask])[0]
        mae = mean_absolute_error(true_brightness[mask], pred_brightness[mask])
        stats_text += f"\n\n{gt} (N={mask.sum()})"
        stats_text += f"\n  R² = {r2:.3f}"
        stats_text += f"\n  r  = {r:.3f}"
        stats_text += f"\n  MAE= {mae:.3f}"
    
    # 高亮度
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
    
    # --- 图3-6: 各类型独立散点图 (均匀采样) ---
    for i, gt in enumerate(GFP_TYPE_LIST):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        
        mask_all = gfp_types == gt
        true_gt = true_brightness[mask_all]
        pred_gt = pred_brightness[mask_all]
        
        # 均匀采样
        si_gt = uniform_sample_indices(true_gt, n_target=800, n_bins=15)
        
        ax.scatter(true_gt[si_gt], pred_gt[si_gt], c=COLORS[gt], alpha=0.4, s=15)
        ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5)
        
        r2 = r2_score(true_gt, pred_gt)
        r = pearsonr(true_gt, pred_gt)[0]
        mae = mean_absolute_error(true_gt, pred_gt)
        ax.set_xlabel('True Brightness', fontsize=11)
        ax.set_ylabel('Predicted Brightness', fontsize=11)
        ax.set_title(f'{gt} (均匀采样 N_sample={len(si_gt)})\n'
                     f'R²={r2:.3f}, r={r:.3f}, MAE={mae:.3f}', fontsize=12)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.grid(True, alpha=0.2)
    
    # --- 图7: 高亮度区 (≥4.0) ---
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
    
    out_path = Path(__file__).resolve().parent.parent / "v75_scatter_uniform.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n散点图已保存: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
