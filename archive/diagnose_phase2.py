"""快速诊断Phase 2.5各模型预测值分布"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from src.config import LOCAL_ESM2_DIR, MODELS_DIR, RAW_DIR, WT_AVGFP, get_device
from src.phase2_generate_hierarchical import (
    parse_mutations, ResidualBrightnessMLP, ESM2LoRARegressorV5, ESM2LoRARegressorV9,
    extract_esm2_embeddings, predict_v7, predict_v5, predict_v9,
    GFP_TYPE_TO_IDX, GFP_WT_BRIGHTNESS, HIGH_BRIGHTNESS_TYPES,
    CHROMOPHORE_AND_CORE_INDICES, ALL_AA
)
import joblib, re

device = get_device()
print(f"Device: {device}")

# 1. 取几条高亮度父本
raw_df = pd.read_excel(RAW_DIR / "GFP_data.xlsx", sheet_name="brightness")
cgre_top = raw_df[raw_df['GFP type'].str.contains('cgreGFP', case=False, na=False)].nlargest(5, 'Brightness')
print(f"\ncgreGFP Top5 亮度: {cgre_top['Brightness'].values}")

# 解析序列
seqs = []
types = []
type_idxs = []
for _, row in cgre_top.iterrows():
    seq = parse_mutations(row['aaMutations'], WT_AVGFP)
    if seq:
        seqs.append(seq)
        types.append('cgreGFP')
        type_idxs.append(GFP_TYPE_TO_IDX['cgreGFP'])

# 生成几条突变体
rng = np.random.default_rng(42)
mutable = [i for i in range(len(WT_AVGFP)) if i not in CHROMOPHORE_AND_CORE_INDICES]
mut_seqs = []
mut_types = []
mut_type_idxs = []
for i, seq in enumerate(seqs):
    for _ in range(10):
        s = list(seq)
        positions = rng.choice(mutable, 3, replace=False)
        for pos in positions:
            alts = [aa for aa in ALL_AA if aa != s[pos]]
            s[pos] = rng.choice(alts)
        mut_seqs.append("".join(s))
        mut_types.append('cgreGFP')
        mut_type_idxs.append(GFP_TYPE_TO_IDX['cgreGFP'])

all_seqs = seqs + mut_seqs
all_type_idxs = np.array(type_idxs + mut_type_idxs)
print(f"\n总序列数: {len(all_seqs)} (父本{len(seqs)} + 突变体{len(mut_seqs)})")

# 2. 提取ESM-2嵌入
print("\n=== 加载ESM-2 ===")
tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
base_model = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR)).to(device)
base_model.eval()

print("=== 提取嵌入 ===")
embeddings = extract_esm2_embeddings(base_model, tokenizer, device, all_seqs, batch_size=16)

# 3. V7预测
print("\n=== V7预测 ===")
model_dirs = sorted(MODELS_DIR.glob("ESM2_ResidualMLP_V75_72k_*"))
model_dir = model_dirs[-1]
print(f"  模型: {model_dir.name}")
config = joblib.load(model_dir / "model_config.joblib")
v7 = ResidualBrightnessMLP(input_dim=config.get('input_dim', 1280))
v7.load_state_dict(torch.load(model_dir / "mlp_model.pt", map_location=device, weights_only=False))
v7.to(device); v7.eval()
pred_v7 = predict_v7(v7, device, embeddings, all_type_idxs)
print(f"  V7预测: min={pred_v7.min():.4f}, max={pred_v7.max():.4f}, mean={pred_v7.mean():.4f}")
print(f"  父本V7: {pred_v7[:len(seqs)]}")
print(f"  突变体V7: min={pred_v7[len(seqs):].min():.4f}, max={pred_v7[len(seqs):].max():.4f}")
print(f"  V7>=4.2: {(pred_v7>=4.2).sum()}/{len(pred_v7)}")
print(f"  V7>=4.8: {(pred_v7>=4.8).sum()}/{len(pred_v7)}")

# 4. V9预测
print("\n=== V9预测 ===")
v9_dirs = sorted(MODELS_DIR.glob("ESM2_LoRA_V9_72k_*"))
v9_dir = v9_dirs[-1]
print(f"  模型: {v9_dir.name}")
v9_tok = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
v9_base = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR))
v9_lora = PeftModel.from_pretrained(v9_base, v9_dir)
v9_merged = v9_lora.merge_and_unload()
v9 = ESM2LoRARegressorV9(v9_merged)
head_state = torch.load(v9_dir / "regressor.pt", map_location=device, weights_only=False)
v9.regressor.load_state_dict(head_state)
v9.to(device); v9.eval()
pred_v9_delta = predict_v9(v9_tok, v9, device, all_seqs, all_type_idxs, batch_size=16)
pred_v9_abs = pred_v9_delta + np.array([GFP_WT_BRIGHTNESS[t] for t in types + mut_types])
print(f"  V9 delta: min={pred_v9_delta.min():.4f}, max={pred_v9_delta.max():.4f}")
print(f"  V9 abs: min={pred_v9_abs.min():.4f}, max={pred_v9_abs.max():.4f}, mean={pred_v9_abs.mean():.4f}")
print(f"  父本V9_abs: {pred_v9_abs[:len(seqs)]}")
print(f"  V9_abs>=4.2: {(pred_v9_abs>=4.2).sum()}/{len(pred_v9_abs)}")
print(f"  V9_abs>=4.8: {(pred_v9_abs>=4.8).sum()}/{len(pred_v9_abs)}")

# 5. V5预测
print("\n=== V5预测 ===")
v5_dirs = list(MODELS_DIR.glob("ESM2_LoRA_V5_TypeEmbed_*"))
v5_dir = max(v5_dirs, key=lambda p: p.stat().st_mtime)
print(f"  模型: {v5_dir.name}")
v5_tok = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
v5_base = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR)).to(device)
v5_base.eval()
v5_peft = PeftModel.from_pretrained(v5_base, str(v5_dir))
v5 = ESM2LoRARegressorV5(v5_peft).to(device)
v5.regressor.load_state_dict(torch.load(v5_dir / "regressor.pt", map_location=device, weights_only=False))
v5.eval()
pred_v5_delta = predict_v5(v5_tok, v5, device, all_seqs, all_type_idxs, batch_size=16)
pred_v5_abs = pred_v5_delta + np.array([GFP_WT_BRIGHTNESS[t] for t in types + mut_types])
print(f"  V5 delta: min={pred_v5_delta.min():.4f}, max={pred_v5_delta.max():.4f}")
print(f"  V5 abs: min={pred_v5_abs.min():.4f}, max={pred_v5_abs.max():.4f}, mean={pred_v5_abs.mean():.4f}")
print(f"  父本V5_abs: {pred_v5_abs[:len(seqs)]}")
print(f"  V5_abs>=4.2: {(pred_v5_abs>=4.2).sum()}/{len(pred_v5_abs)}")
print(f"  V5_abs>=4.8: {(pred_v5_abs>=4.8).sum()}/{len(pred_v5_abs)}")

# 6. 加权平均
print("\n=== 加权平均 (V5=0.15, V7=0.35, V9=0.50) ===")
w_avg = 0.15 * pred_v5_abs + 0.35 * pred_v7 + 0.50 * pred_v9_abs
print(f"  加权平均: min={w_avg.min():.4f}, max={w_avg.max():.4f}, mean={w_avg.mean():.4f}")
print(f"  父本加权平均: {w_avg[:len(seqs)]}")
print(f"  >=4.8: {(w_avg>=4.8).sum()}/{len(w_avg)}")
print(f"  >=4.6: {(w_avg>=4.6).sum()}/{len(w_avg)}")
print(f"  >=4.4: {(w_avg>=4.4).sum()}/{len(w_avg)}")

# 共识
votes = ((pred_v5_abs>=4.8).astype(int) + (pred_v7>=4.8).astype(int) + (pred_v9_abs>=4.8).astype(int))
print(f"\n  共识票数分布: 0票={sum(votes==0)}, 1票={sum(votes==1)}, 2票={sum(votes==2)}, 3票={sum(votes==3)}")

print("\n=== 逐条对比 ===")
print(f"{'序列':>6} {'类型':>8} {'V5_abs':>8} {'V7_abs':>8} {'V9_abs':>8} {'加权平均':>8} {'票数':>4}")
for i in range(min(20, len(all_seqs))):
    t = types[i] if i < len(seqs) else mut_types[i-len(seqs)]
    v = (pred_v5_abs[i]>=4.8) + (pred_v7[i]>=4.8) + (pred_v9_abs[i]>=4.8)
    print(f"  {i:4d} {t:>8} {pred_v5_abs[i]:8.4f} {pred_v7[i]:8.4f} {pred_v9_abs[i]:8.4f} {w_avg[i]:8.4f} {v:4d}")
