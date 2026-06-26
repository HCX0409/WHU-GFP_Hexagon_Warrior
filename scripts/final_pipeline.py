"""
终极数据汇总: 合并所有历史数据 → 统一三模型打分 → 选最终5条
策略: 每类型至少1条 + 1条给最强类型
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import gc
from collections import defaultdict

from src.config import GEN_SEQS_DIR, AUTHOR, DATE_STR, get_device
from src.phase2_generate_hierarchical import (
    ALL_GFP_TYPES, GFP_TYPE_TO_IDX, TYPE_CONSENSUS_TARGET,
    MODEL_WEIGHTS, CONSENSUS_VOTES, GFP_WT_BRIGHTNESS,
    load_v7_model, predict_v7, extract_esm2_embeddings,
    load_v9_model, predict_v9, load_v5_model, predict_v5,
    clear_gpu_cache
)

# ==================== Step 1: 合并所有数据源 ====================
print("="*60)
print(" Step 1: 合并所有数据源并去重")
print("="*60)

all_records = []
sources = {}

# --- Phase2.5 full_scored (已有三模型打分) ---
df = pd.read_csv(GEN_SEQS_DIR / 'phase2_full_scored_HCX_260623.csv')
for _, row in df.iterrows():
    all_records.append({
        'sequence': row['sequence'],
        'gfp_type': row['gfp_type'],
        'n_mutations': row.get('n_mutations', 3),
        'parent_brightness': row.get('parent_brightness', np.nan),
        'pred_v7': row['pred_v7'],
        'pred_v9_abs': row['pred_v9_abs'],
        'pred_v5_abs': row['pred_v5_abs'],
        'weighted_avg': row['weighted_avg'],
        'votes': row.get('votes', 0),
        'has_scores': True,
        'source': 'phase2.5',
    })
sources['phase2.5'] = len(df)

# --- Phase2 crossval (有打分但列名不同) ---
df = pd.read_csv(GEN_SEQS_DIR / 'intermediate' / 'phase2_crossval_HCX_260620.csv')
for _, row in df.iterrows():
    all_records.append({
        'sequence': row['sequence'],
        'gfp_type': row['gfp_type'],
        'n_mutations': row.get('num_mutations_from_parent', 0),
        'parent_brightness': row.get('parent_brightness', np.nan),
        'pred_v7': row['pred_v7'],
        'pred_v9_abs': row['pred_v9_abs'],
        'pred_v5_abs': row['pred_v5'],  # 列名不同
        'weighted_avg': row.get('weighted_avg', 0),
        'votes': 0,
        'has_scores': True,
        'source': 'crossval',
    })
sources['crossval'] = len(df)

# --- Phase4 Top20 (需要重打分) ---
df = pd.read_csv(GEN_SEQS_DIR / 'intermediate' / 'phase4_final_top20_HCX_260613.csv')
for _, row in df.iterrows():
    all_records.append({
        'sequence': row['sequence'],
        'gfp_type': row['gfp_type'],
        'n_mutations': row.get('num_mutations_from_parent', 0),
        'parent_brightness': row.get('parent_brightness', np.nan),
        'pred_v7': np.nan,
        'pred_v9_abs': np.nan,
        'pred_v5_abs': np.nan,
        'weighted_avg': np.nan,
        'votes': 0,
        'has_scores': False,
        'source': 'phase4_top20',
    })
sources['phase4_top20'] = len(df)

# --- Phase2探索性扫描 (32394条, 需重打分) ---
df = pd.read_csv(GEN_SEQS_DIR / 'archive' / 'phase2_bright_mut_5000_HCX_260611.csv')
for _, row in df.iterrows():
    all_records.append({
        'sequence': row['sequence'],
        'gfp_type': row['gfp_type'],
        'n_mutations': row.get('num_mutations_from_parent', 0),
        'parent_brightness': row.get('parent_brightness', np.nan),
        'pred_v7': np.nan,
        'pred_v9_abs': np.nan,
        'pred_v5_abs': np.nan,
        'weighted_avg': np.nan,
        'votes': 0,
        'has_scores': False,
        'source': 'phase2_exploratory',
    })
sources['phase2_exploratory'] = len(df)

big_df = pd.DataFrame(all_records)
print(f"  各来源: {sources}")
print(f"  合计: {len(big_df)}条")

# 去重: 保留有分数的版本
big_df = big_df.sort_values('has_scores', ascending=False)
big_df = big_df.drop_duplicates(subset=['sequence'], keep='first').reset_index(drop=True)
print(f"  去重后: {len(big_df)}条")

need_score = ~big_df['has_scores']
n_need = need_score.sum()
n_have = (~need_score).sum()
print(f"  已有分数: {n_have}条, 需要打分: {n_need}条")

for gt in ALL_GFP_TYPES:
    n = (big_df['gfp_type'] == gt).sum()
    n_ns = (big_df.loc[need_score, 'gfp_type'] == gt).sum()
    print(f"    {gt}: {n}条 (需打分:{n_ns})")

# ==================== Step 2: 对缺失分数的序列统一打分 ====================
if n_need > 0:
    print(f"\n{'='*60}")
    print(f" Step 2: 三模型统一打分 ({n_need}条)")
    print(f"{'='*60}")
    
    need_df = big_df[need_score].copy()
    need_seqs = need_df['sequence'].tolist()
    need_type_idx = need_df['gfp_type'].map(GFP_TYPE_TO_IDX).values
    
    # V7 (MLP, 快)
    print("\n  加载V7 + ESM-2...")
    v7_model, v7_device = load_v7_model()
    from src.phase2_generate_hierarchical import load_esm2_base_for_embeddings
    esm2_base, esm2_tok, esm2_dev = load_esm2_base_for_embeddings()
    
    print(f"  ESM-2嵌入提取 ({n_need}条)...")
    embeddings = extract_esm2_embeddings(esm2_base, esm2_tok, esm2_dev, need_seqs)
    pred_v7 = predict_v7(v7_model, v7_device, embeddings, need_type_idx)
    
    del esm2_base, esm2_tok, v7_model
    clear_gpu_cache()
    print(f"  V7完成: mean={pred_v7.mean():.3f}")
    
    # V9
    print("\n  加载V9...")
    v9_tok, v9_model, v9_dev = load_v9_model()
    pred_v9_delta = predict_v9(v9_tok, v9_model, v9_dev, need_seqs, need_type_idx)
    pred_v9_abs = np.array([
        d + GFP_WT_BRIGHTNESS[t] for d, t in zip(pred_v9_delta, need_df['gfp_type'].values)
    ])
    del v9_tok, v9_model
    clear_gpu_cache()
    print(f"  V9完成: abs mean={pred_v9_abs.mean():.3f}")
    
    # V5
    print("\n  加载V5...")
    v5_tok, v5_model, v5_dev = load_v5_model()
    pred_v5_delta = predict_v5(v5_tok, v5_model, v5_dev, need_seqs, need_type_idx)
    pred_v5_abs = np.array([
        d + GFP_WT_BRIGHTNESS[t] for d, t in zip(pred_v5_delta, need_df['gfp_type'].values)
    ])
    del v5_tok, v5_model
    clear_gpu_cache()
    print(f"  V5完成: abs mean={pred_v5_abs.mean():.3f}")
    
    # 写回big_df
    w = MODEL_WEIGHTS
    weighted = w['v5']*pred_v5_abs + w['v7']*pred_v7 + w['v9']*pred_v9_abs
    
    need_indices = need_df.index
    big_df.loc[need_indices, 'pred_v7'] = pred_v7
    big_df.loc[need_indices, 'pred_v9_abs'] = pred_v9_abs
    big_df.loc[need_indices, 'pred_v5_abs'] = pred_v5_abs
    big_df.loc[need_indices, 'weighted_avg'] = weighted
else:
    print("\n  所有序列已有分数, 跳过打分")

# ==================== Step 3: 重算votes ====================
print(f"\n{'='*60}")
print(f" Step 3: 共识筛选 (按类型独立阈值)")
print(f"{'='*60}")

big_df['votes'] = 0
for gt in ALL_GFP_TYPES:
    target = TYPE_CONSENSUS_TARGET[gt]
    mask = big_df['gfp_type'] == gt
    sub = big_df[mask]
    v = np.zeros(len(sub), dtype=int)
    v += (sub['pred_v5_abs'].values >= target).astype(int)
    v += (sub['pred_v7'].values >= target).astype(int)
    v += (sub['pred_v9_abs'].values >= target).astype(int)
    big_df.loc[mask, 'votes'] = v

# ==================== Step 4: 按类型选Top候选 ====================
print(f"\n{'='*60}")
print(f" Step 4: 按类型选Top候选")
print(f"{'='*60}")

# 先按类型看共识通过情况
for gt in ALL_GFP_TYPES:
    target = TYPE_CONSENSUS_TARGET[gt]
    sub = big_df[(big_df['gfp_type'] == gt) & (big_df['votes'] >= CONSENSUS_VOTES)]
    sub = sub.sort_values('weighted_avg', ascending=False)
    print(f"\n  {gt} (目标≥{target}): 共识通过{len(sub)}条")
    if len(sub) > 0:
        top5 = sub.head(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            print(f"    #{i+1}: weighted={row['weighted_avg']:.3f}, "
                  f"V7={row['pred_v7']:.3f}, V9={row['pred_v9_abs']:.3f}, V5={row['pred_v5_abs']:.3f}, "
                  f"votes={row['votes']:.0f}, muts={row['n_mutations']:.0f}, "
                  f"source={row['source']}")

# ==================== Step 5: 选最终5条 ====================
print(f"\n{'='*60}")
print(f" Step 5: 选最终5条 (每类型≥1 + 1条最强)")
print(f"{'='*60}")

final_5 = []
used_seqs = set()

# 每类型选最佳1条
for gt in ALL_GFP_TYPES:
    sub = big_df[(big_df['gfp_type'] == gt) & (big_df['votes'] >= CONSENSUS_VOTES)]
    sub = sub.sort_values('weighted_avg', ascending=False)
    if len(sub) > 0:
        best = sub.iloc[0]
        final_5.append(best)
        used_seqs.add(best['sequence'])
        print(f"  {gt}: weighted={best['weighted_avg']:.3f}, votes={best['votes']:.0f}")
    else:
        # 无共识通过的，取weighted_avg最高的
        sub = big_df[big_df['gfp_type'] == gt].sort_values('weighted_avg', ascending=False)
        if len(sub) > 0:
            best = sub.iloc[0]
            final_5.append(best)
            used_seqs.add(best['sequence'])
            print(f"  {gt}: (无共识) weighted={best['weighted_avg']:.3f}, votes={best['votes']:.0f}")

# 第5条: 跨类型取剩余最高weighted_avg
remaining = big_df[~big_df['sequence'].isin(used_seqs)].sort_values('weighted_avg', ascending=False)
if len(remaining) > 0:
    # 优先从共识通过的中选
    remaining_consensus = remaining[remaining['votes'] >= CONSENSUS_VOTES]
    if len(remaining_consensus) > 0:
        best = remaining_consensus.iloc[0]
    else:
        best = remaining.iloc[0]
    final_5.append(best)
    used_seqs.add(best['sequence'])
    print(f"  第5条({best['gfp_type']}): weighted={best['weighted_avg']:.3f}, votes={best['votes']:.0f}")

final_df = pd.DataFrame(final_5)
print(f"\n最终5条:")
for i, (_, row) in enumerate(final_df.iterrows()):
    print(f"  #{i+1} [{row['gfp_type']}]: weighted={row['weighted_avg']:.3f}, "
          f"V7={row['pred_v7']:.3f}, V9={row['pred_v9_abs']:.3f}, V5={row['pred_v5_abs']:.3f}, "
          f"votes={row['votes']:.0f}, muts={row['n_mutations']:.0f}")

# 保存
out_path = GEN_SEQS_DIR / f"final_5_candidates_{AUTHOR}_{DATE_STR}.csv"
out_cols = ['sequence', 'gfp_type', 'n_mutations', 'parent_brightness',
            'pred_v7', 'pred_v9_abs', 'pred_v5_abs', 'weighted_avg', 'votes', 'source']
out_cols = [c for c in out_cols if c in final_df.columns]
final_df[out_cols].to_csv(out_path, index=False)
print(f"\n保存: {out_path.name}")

# 也保存完整打分数据
full_path = GEN_SEQS_DIR / f"all_scored_merged_{AUTHOR}_{DATE_STR}.csv"
all_cols = [c for c in out_cols if c in big_df.columns]
big_df.sort_values('weighted_avg', ascending=False)[all_cols].to_csv(full_path, index=False)
print(f"完整数据: {full_path.name} ({len(big_df)}条)")
