"""从已保存的iter4+iter5数据重跑V9/V5打分+共识筛选"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import torch
import gc

from src.config import GEN_SEQS_DIR, AUTHOR, DATE_STR, WT_AVGFP, RANDOM_SEED, get_device
from src.phase2_generate_hierarchical import (
    ALL_GFP_TYPES, GFP_TYPE_TO_IDX, TYPE_CONSENSUS_TARGET, V7_FILTER_PER_TYPE,
    MODEL_WEIGHTS, CONSENSUS_VOTES, GFP_WT_BRIGHTNESS,
    load_v9_model, predict_v9, load_v5_model, predict_v5,
    clear_gpu_cache
)

# ========== 加载所有迭代数据 ==========
all_dfs = []
for f in sorted(GEN_SEQS_DIR.glob("phase2_iter*_HCX_*.csv")):
    df = pd.read_csv(f)
    print(f"  {f.name}: {len(df)}条")
    all_dfs.append(df)

stage1_df = pd.concat(all_dfs, ignore_index=True)
stage1_df = stage1_df.sort_values('pred_v7', ascending=False).drop_duplicates(
    subset=['sequence'], keep='first').reset_index(drop=True)
print(f"\n合并去重后: {len(stage1_df)}条")
for gt in ALL_GFP_TYPES:
    n = (stage1_df['gfp_type'] == gt).sum()
    print(f"  {gt}: {n}条")

# ========== V9打分 ==========
print(f"\n{'='*60}")
print(f" Stage 2a: V9 LoRA精筛 ({len(stage1_df)}条)")
print(f"{'='*60}")

stage1_seqs = stage1_df['sequence'].tolist()
stage1_type_idx = stage1_df['gfp_type'].map(GFP_TYPE_TO_IDX).values

v9_tokenizer, v9_model, v9_device = load_v9_model()
pred_v9_delta = predict_v9(v9_tokenizer, v9_model, v9_device, stage1_seqs, stage1_type_idx)
pred_v9_abs = np.array([
    d + GFP_WT_BRIGHTNESS[t] for d, t in zip(pred_v9_delta, stage1_df['gfp_type'].values)
])
stage1_df['pred_v9_delta'] = pred_v9_delta
stage1_df['pred_v9_abs'] = pred_v9_abs
print(f"  V9完成: delta均值={pred_v9_delta.mean():.3f}, abs均值={pred_v9_abs.mean():.3f}")

del v9_tokenizer, v9_model
clear_gpu_cache()

# ========== V5打分 ==========
print(f"\n{'='*60}")
print(f" Stage 2b: V5 LoRA精筛 ({len(stage1_df)}条)")
print(f"{'='*60}")

v5_tokenizer, v5_model, v5_device = load_v5_model()
pred_v5_delta = predict_v5(v5_tokenizer, v5_model, v5_device, stage1_seqs, stage1_type_idx)
pred_v5_abs = np.array([
    d + GFP_WT_BRIGHTNESS[t] for d, t in zip(pred_v5_delta, stage1_df['gfp_type'].values)
])
stage1_df['pred_v5_delta'] = pred_v5_delta
stage1_df['pred_v5_abs'] = pred_v5_abs
print(f"  V5完成: delta均值={pred_v5_delta.mean():.3f}, abs均值={pred_v5_abs.mean():.3f}")

del v5_tokenizer, v5_model
clear_gpu_cache()

# ========== Stage 3: 共识筛选 ==========
print(f"\n{'='*60}")
print(f" Stage 3: 三模型共识筛选")
print(f"{'='*60}")

w = MODEL_WEIGHTS
stage1_df['weighted_avg'] = (
    w['v5'] * stage1_df['pred_v5_abs'] +
    w['v7'] * stage1_df['pred_v7'] +
    w['v9'] * stage1_df['pred_v9_abs']
)

stage1_df['votes'] = 0
stage1_df['type_target'] = 0.0
for gt in ALL_GFP_TYPES:
    target = TYPE_CONSENSUS_TARGET[gt]
    mask = stage1_df['gfp_type'] == gt
    stage1_df.loc[mask, 'type_target'] = target
    votes = np.zeros(mask.sum(), dtype=int)
    sub = stage1_df[mask]
    votes += (sub['pred_v5_abs'].values >= target).astype(int)
    votes += (sub['pred_v7'].values >= target).astype(int)
    votes += (sub['pred_v9_abs'].values >= target).astype(int)
    stage1_df.loc[mask, 'votes'] = votes

consensus_df = stage1_df[stage1_df['votes'] >= CONSENSUS_VOTES].copy()
consensus_df = consensus_df.sort_values('weighted_avg', ascending=False).reset_index(drop=True)

print(f"\n共识通过 (votes>={CONSENSUS_VOTES}): {len(consensus_df)}条")
for gt in ALL_GFP_TYPES:
    sub = consensus_df[consensus_df['gfp_type'] == gt]
    if len(sub) > 0:
        print(f"  {gt}: {len(sub)}条, weighted_avg=[{sub['weighted_avg'].min():.3f}, {sub['weighted_avg'].max():.3f}]")
    else:
        print(f"  {gt}: 0条")

# ========== 保存 ==========
out_cols = ['sequence', 'gfp_type', 'n_mutations', 'parent_rank', 'parent_brightness',
            'pred_v7', 'pred_v9_abs', 'pred_v5_abs', 'weighted_avg', 'votes']
out_cols = [c for c in out_cols if c in consensus_df.columns]

out_path = GEN_SEQS_DIR / f"phase2_hierarchical_{AUTHOR}_{DATE_STR}.csv"
consensus_df[out_cols].to_csv(out_path, index=False)
print(f"\n最终结果: {out_path.name} ({len(consensus_df)}条)")

# 也保存完整打分数据
full_path = GEN_SEQS_DIR / f"phase2_full_scored_{AUTHOR}_{DATE_STR}.csv"
full_cols = [c for c in out_cols if c in stage1_df.columns]
stage1_df.sort_values('weighted_avg', ascending=False)[full_cols].to_csv(full_path, index=False)
print(f"完整打分: {full_path.name} ({len(stage1_df)}条)")
