"""
V6模型散点图: 真实亮度 vs 预测亮度
重点关注 cgreGFP 和 ppluGFP 在高亮度区(brightness>=4.0)的表现
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

from src.config import LOCAL_ESM2_DIR, MODELS_DIR, PROCESSED_DIR, get_device

# ======================== 常量 ========================
GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {t: i for i, t in enumerate(GFP_TYPE_LIST)}

WT_BRIGHTNESS = {
    'avGFP': 3.7192,
    'amacGFP': 3.9707,
    'cgreGFP': 4.4969,
    'ppluGFP': 4.2258
}

MODEL_DIR = MODELS_DIR / "ESM2_LoRA_V6_HiBright_HCX_260618"


# ======================== 模型定义 ========================
class ESM2LoRARegressorV6(nn.Module):
    def __init__(self, base_model, num_gfp_types=4):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        input_dim = hidden_size + num_gfp_types
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 768), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.num_gfp_types = num_gfp_types

    def forward(self, input_ids, attention_mask, gfp_type_idx):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        type_onehot = torch.zeros(
            gfp_type_idx.size(0), self.num_gfp_types, device=gfp_type_idx.device
        )
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        combined = torch.cat([pooled, type_onehot], dim=1)
        return self.regressor(combined).squeeze(-1)


class InferenceDataset(Dataset):
    def __init__(self, seqs, type_indices, tokenizer, max_len=250):
        self.seqs = seqs
        self.type_indices = type_indices
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
            "gfp_type": torch.tensor(self.type_indices[idx], dtype=torch.long)
        }


# ======================== 推理函数 ========================
@torch.no_grad()
def predict(model, tokenizer, device, seqs, type_indices, batch_size=64):
    dataset = InferenceDataset(seqs, type_indices, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    preds = []
    model.eval()
    for batch in tqdm(loader, desc="推理中"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gfp_type = batch["gfp_type"].to(device)
        
        pred = model(input_ids, attention_mask, gfp_type)
        preds.extend(pred.cpu().numpy())
    
    return np.array(preds)


# ======================== 主函数 ========================
def main():
    device = get_device()
    print(f"Device: {device}")
    
    # 1. 加载数据
    print("=" * 60)
    print("加载数据...")
    csv_path = PROCESSED_DIR / "v6_training_data.csv"
    df = pd.read_csv(csv_path)
    print(f"  总样本数: {len(df)}")
    
    seqs = df['sequence'].tolist()
    gfp_types = df['gfp_type'].tolist()
    true_brightness = df['brightness'].values
    type_indices = [GFP_TYPE_TO_IDX[gt] for gt in gfp_types]
    
    # 2. 加载模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModel.from_pretrained(LOCAL_ESM2_DIR)
    
    lora_config = LoraConfig.from_pretrained(MODEL_DIR)
    peft_model = PeftModel(base_model, lora_config)
    
    model = ESM2LoRARegressorV6(peft_model).to(device)
    model.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    regressor_path = MODEL_DIR / "regressor.pt"
    model.regressor.load_state_dict(
        torch.load(regressor_path, map_location=device, weights_only=True)
    )
    model.eval()
    print("  模型加载完成")
    
    # 3. 推理
    print("=" * 60)
    pred_delta = predict(model, tokenizer, device, seqs, type_indices)
    
    # delta -> 绝对亮度
    wt_brightness = np.array([WT_BRIGHTNESS[gt] for gt in gfp_types])
    pred_brightness = pred_delta + wt_brightness
    
    # 4. 整体指标
    print("=" * 60)
    print(f"整体: R²={r2_score(true_brightness, pred_brightness):.4f}, "
          f"r={pearsonr(true_brightness, pred_brightness)[0]:.4f}, "
          f"MAE={mean_absolute_error(true_brightness, pred_brightness):.4f}")
    
    # 5. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    colors = {
        'avGFP': '#2196F3',
        'amacGFP': '#4CAF50', 
        'cgreGFP': '#FF9800',
        'ppluGFP': '#9C27B0'
    }
    
    # --- 图1: 全部4类型 ---
    ax = axes[0, 0]
    for gt in GFP_TYPE_LIST:
        mask = df['gfp_type'] == gt
        ax.scatter(true_brightness[mask], pred_brightness[mask], 
                   c=colors[gt], alpha=0.3, s=8, label=gt)
    lims = [min(true_brightness.min(), pred_brightness.min()) - 0.2,
            max(true_brightness.max(), pred_brightness.max()) + 0.2]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('True Brightness', fontsize=11)
    ax.set_ylabel('Predicted Brightness', fontsize=11)
    ax.set_title('V6: All GFP Types (True vs Predicted)', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    
    # --- 图2: cgreGFP 全部 ---
    ax = axes[0, 1]
    mask_cg = df['gfp_type'] == 'cgreGFP'
    true_cg = true_brightness[mask_cg]
    pred_cg = pred_brightness[mask_cg]
    ax.scatter(true_cg, pred_cg, c=colors['cgreGFP'], alpha=0.3, s=10)
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    r2 = r2_score(true_cg, pred_cg)
    r = pearsonr(true_cg, pred_cg)[0]
    mae = mean_absolute_error(true_cg, pred_cg)
    ax.set_xlabel('True Brightness', fontsize=11)
    ax.set_ylabel('Predicted Brightness', fontsize=11)
    ax.set_title(f'cgreGFP (All, N={mask_cg.sum()})\nR²={r2:.3f}, r={r:.3f}, MAE={mae:.3f}', fontsize=12)
    ax.set_xlim(lims); ax.set_ylim(lims)
    
    # --- 图3: ppluGFP 全部 ---
    ax = axes[1, 0]
    mask_pp = df['gfp_type'] == 'ppluGFP'
    true_pp = true_brightness[mask_pp]
    pred_pp = pred_brightness[mask_pp]
    ax.scatter(true_pp, pred_pp, c=colors['ppluGFP'], alpha=0.3, s=10)
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    r2 = r2_score(true_pp, pred_pp)
    r = pearsonr(true_pp, pred_pp)[0]
    mae = mean_absolute_error(true_pp, pred_pp)
    ax.set_xlabel('True Brightness', fontsize=11)
    ax.set_ylabel('Predicted Brightness', fontsize=11)
    ax.set_title(f'ppluGFP (All, N={mask_pp.sum()})\nR²={r2:.3f}, r={r:.3f}, MAE={mae:.3f}', fontsize=12)
    ax.set_xlim(lims); ax.set_ylim(lims)
    
    # --- 图4: 高亮度区(brightness>=4.0) cgreGFP + ppluGFP ---
    ax = axes[1, 1]
    hi_mask_cg = (df['gfp_type'] == 'cgreGFP') & (df['brightness'] >= 4.0)
    hi_mask_pp = (df['gfp_type'] == 'ppluGFP') & (df['brightness'] >= 4.0)
    
    true_hi_cg = true_brightness[hi_mask_cg]
    pred_hi_cg = pred_brightness[hi_mask_cg]
    true_hi_pp = true_brightness[hi_mask_pp]
    pred_hi_pp = pred_brightness[hi_mask_pp]
    
    ax.scatter(true_hi_cg, pred_hi_cg, c=colors['cgreGFP'], alpha=0.3, s=12, 
               label=f'cgreGFP (N={hi_mask_cg.sum()})')
    ax.scatter(true_hi_pp, pred_hi_pp, c=colors['ppluGFP'], alpha=0.3, s=12,
               label=f'ppluGFP (N={hi_mask_pp.sum()})')
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    
    all_hi_true = np.concatenate([true_hi_cg, true_hi_pp])
    all_hi_pred = np.concatenate([pred_hi_cg, pred_hi_pp])
    if len(all_hi_true) > 2:
        r2_hi = r2_score(all_hi_true, all_hi_pred)
        r_hi = pearsonr(all_hi_true, all_hi_pred)[0]
        mae_hi = mean_absolute_error(all_hi_true, all_hi_pred)
        ax.set_title(f'High Brightness (≥4.0) cgre+pplu (N={len(all_hi_true)})\n'
                     f'R²={r2_hi:.3f}, r={r_hi:.3f}, MAE={mae_hi:.3f}', fontsize=12)
    else:
        ax.set_title('High Brightness (≥4.0) cgre+pplu', fontsize=12)
    
    ax.set_xlabel('True Brightness', fontsize=11)
    ax.set_ylabel('Predicted Brightness', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    
    plt.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / "v6_scatter_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n散点图已保存: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
