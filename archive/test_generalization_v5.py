"""
测试V5模型（相对亮度+GFP类型嵌入）在4种GFP类型上的泛化能力
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from src.config import MODELS_DIR, LOCAL_ESM2_DIR, get_device, WT_AVGFP
import warnings
warnings.filterwarnings('ignore')

GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {t: i for i, t in enumerate(GFP_TYPE_LIST)}

WT_BRIGHTNESS = {'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258}


class ESM2LoRARegressorV5(nn.Module):
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
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        type_onehot = torch.zeros(pooled.size(0), self.num_gfp_types, device=pooled.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        combined = torch.cat([pooled, type_onehot], dim=1)
        return self.regressor(combined).squeeze(-1)


def load_v5_model():
    print("="*60)
    print("🔧 正在加载V5模型（相对亮度+类型嵌入）...")
    print("="*60)
    
    device = get_device()
    
    model_dirs = list(MODELS_DIR.glob("ESM2_LoRA_V5_TypeEmbed_*"))
    if not model_dirs:
        raise FileNotFoundError("❌ 未找到V5模型！请先运行phase1_lora_finetune_v5.py")
    
    latest_model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
    print(f"加载模型: {latest_model_dir.name}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    base_model = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR)).to(device)
    base_model.eval()
    
    peft_model = PeftModel.from_pretrained(base_model, str(latest_model_dir))
    model = ESM2LoRARegressorV5(peft_model, num_gfp_types=len(GFP_TYPE_LIST)).to(device)
    
    regressor_path = latest_model_dir / "regressor.pt"
    if regressor_path.exists():
        model.regressor.load_state_dict(torch.load(regressor_path, map_location=device, weights_only=True))
        print("✅ Regressor权重加载成功")
    
    model.eval()
    return tokenizer, model, device


def predict_delta(tokenizer, model, device, sequences, gfp_type, batch_size=48):
    """预测相对亮度变化"""
    type_idx = GFP_TYPE_TO_IDX.get(gfp_type, 0)
    predictions = []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = list(sequences[i:i+batch_size])
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=250).to(device)
        type_tensor = torch.full((len(batch_seqs),), type_idx, dtype=torch.long, device=device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            preds = model(inputs['input_ids'], inputs['attention_mask'], type_tensor)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)


def analyze_gfp_type(raw_df, gfp_type, tokenizer, model, device, n_samples=1000):
    print(f"\n{'='*60}")
    print(f"📊 分析 {gfp_type} ...")
    print(f"{'='*60}")
    
    b_column = raw_df.columns[1]
    mask = raw_df[b_column].astype(str).str.contains(gfp_type, case=False, na=False)
    filtered_df = raw_df[mask].copy()
    
    print(f"原始数据中找到 {len(filtered_df)} 条 {gfp_type}")
    if len(filtered_df) == 0:
        return None
    
    from src.phase1_data import parse_mutations
    filtered_df['full_sequence'] = [parse_mutations(x, WT_AVGFP) for x in filtered_df['aaMutations']]
    filtered_df = filtered_df.dropna(subset=['full_sequence'])
    
    if len(filtered_df) == 0:
        return None
    
    if len(filtered_df) > n_samples:
        sampled_df = filtered_df.sample(n=n_samples, random_state=42)
    else:
        sampled_df = filtered_df
    
    sequences = sampled_df['full_sequence'].tolist()
    true_brightness = sampled_df['Brightness'].values
    wt_bright = WT_BRIGHTNESS.get(gfp_type, 3.7192)
    
    print(f"-> 预测中 (N={len(sequences)}, WT={wt_bright:.2f})...")
    pred_delta = predict_delta(tokenizer, model, device, sequences, gfp_type)
    pred_brightness = pred_delta + wt_bright
    
    r2 = r2_score(true_brightness, pred_brightness)
    mae = mean_absolute_error(true_brightness, pred_brightness)
    
    print(f"\n✅ {gfp_type}: R²={r2:.4f}, MAE={mae:.4f}")
    
    # 散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    min_val = min(true_brightness.min(), pred_brightness.min())
    max_val = max(true_brightness.max(), pred_brightness.max())
    
    ax.scatter(true_brightness, pred_brightness, alpha=0.6, c='steelblue', edgecolors='navy', s=30, linewidth=0.5)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('True Log10 Brightness', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Log10 Brightness', fontsize=12, fontweight='bold')
    ax.set_title(f'{gfp_type} (V5 Type-Embed)\nR²={r2:.4f}, MAE={mae:.4f}, N={len(sequences)}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val - 0.2, max_val + 0.2)
    ax.set_ylim(min_val - 0.2, max_val + 0.2)
    plt.tight_layout()
    
    safe_name = gfp_type.replace(' ', '_')
    output_path = Path(f"model_generalization_v5_{safe_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 散点图: {output_path}")
    plt.close()
    
    return {'gfp_type': gfp_type, 'r2': r2, 'mae': mae, 'n_samples': len(sequences)}


def main():
    print("\n" + "="*60)
    print(" V5模型泛化能力测试（相对亮度+类型嵌入）")
    print("="*60)
    
    tokenizer, model, device = load_v5_model()
    
    raw_df = pd.read_excel("data/raw/GFP_data.xlsx")
    print(f"加载了 {len(raw_df)} 条原始记录")
    
    results = []
    for gfp_type in GFP_TYPE_LIST:
        result = analyze_gfp_type(raw_df, gfp_type, tokenizer, model, device, n_samples=1000)
        if result:
            results.append(result)
    
    print("\n" + "="*60)
    print("🏆 V5泛化能力测试总结")
    print("="*60)
    
    if results:
        summary_df = pd.DataFrame([{
            'GFP类型': r['gfp_type'], '样本数': r['n_samples'],
            'R²': f"{r['r2']:.4f}", 'MAE': f"{r['mae']:.4f}"
        } for r in results])
        print(summary_df.to_string(index=False))
        
        best = max(results, key=lambda x: x['r2'])
        worst = min(results, key=lambda x: x['r2'])
        print(f"\n🎯 最佳: {best['gfp_type']} (R²={best['r2']:.4f})")
        print(f"⚠️  最差: {worst['gfp_type']} (R²={worst['r2']:.4f})")
        print(f"📊 R²差距: {best['r2'] - worst['r2']:.4f}")


if __name__ == "__main__":
    main()
