"""重新挑选cgreGFP和avGFP的替代候选"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pandas as pd
import numpy as np
from src.config import WT_AVGFP as WT

# 加载完整打分数据
all_df = pd.read_csv('generated_seqs/all_scored_merged_HCX_260625.csv')
print(f"全量数据: {len(all_df)}条")

# 加载排除列表
excl_df = pd.read_csv('data/raw/Exclusion_List.csv')
excl_seqs = set(excl_df['Sequence'].values)
print(f"排除列表: {len(excl_seqs)}条")

# 过滤排除列表
in_excl = all_df['sequence'].isin(excl_seqs)
print(f"全量中在排除列表的: {in_excl.sum()}条")
df = all_df[~in_excl].copy()
print(f"过滤后: {len(df)}条")

# 加载当前最终5条
final5 = pd.read_csv('generated_seqs/final_5_candidates_HCX_260625.csv')
exclude_seqs = set(final5['sequence'].values)

# 当前要替换的: cgreGFP(#1) 和 avGFP(#4)
current_cgre = final5[final5['gfp_type']=='cgreGFP']['sequence'].values[0]
current_avgfp = final5[final5['gfp_type']=='avGFP']['sequence'].values[0]
print(f"\n需替换:")
print(f"  cgreGFP: {current_cgre[:50]}... (weighted={final5[final5['sequence']==current_cgre]['weighted_avg'].values[0]:.3f})")
print(f"  avGFP:   {current_avgfp[:50]}... (weighted={final5[final5['sequence']==current_avgfp]['weighted_avg'].values[0]:.3f})")

# 保留的3条 (amacGFP x2 + ppluGFP)
kept = final5[final5['gfp_type'].isin(['amacGFP', 'ppluGFP'])]
print(f"\n保留的3条:")
for _, r in kept.iterrows():
    print(f"  {r['gfp_type']}: weighted={r['weighted_avg']:.3f}")

# 为cgreGFP和avGFP分别找替代
for gfp_type in ['cgreGFP', 'avGFP']:
    print(f"\n{'='*70}")
    print(f"  {gfp_type} 替代候选 Top 10")
    print(f"{'='*70}")
    
    candidates = df[df['gfp_type'] == gfp_type].copy()
    # 排除已有的最终5条
    candidates = candidates[~candidates['sequence'].isin(exclude_seqs)]
    # 按加权平均降序
    candidates = candidates.sort_values('weighted_avg', ascending=False)
    
    print(f"  候选总数: {len(candidates)}条")
    print(f"  加权分数范围: {candidates['weighted_avg'].min():.3f} ~ {candidates['weighted_avg'].max():.3f}")
    
    top10 = candidates.head(10)
    for rank, (_, row) in enumerate(top10.iterrows()):
        seq = row['sequence']
        muts = [(i+1, WT[i], seq[i]) for i in range(min(len(WT), len(seq))) if WT[i] != seq[i]]
        n_muts = len(muts)
        mut_str = ', '.join(f'{a}{p}{b}' for p,a,b in muts[:8])
        if n_muts > 8:
            mut_str += f' +{n_muts-8}more'
        
        votes = row.get('votes', '?')
        v7 = row.get('pred_v7', '?')
        v9 = row.get('pred_v9_abs', '?')
        v5 = row.get('pred_v5_abs', '?')
        source = row.get('source', '?')
        
        print(f"\n  #{rank+1}: weighted={row['weighted_avg']:.3f}, V7={v7}, V9={v9}, V5={v5}")
        print(f"    votes={votes}, {n_muts}个突变, source={source}")
        print(f"    突变: {mut_str}")
        
        # 检查关键结构位点
        chromophore = f"{seq[64]}{seq[65]}{seq[66]}" if len(seq) > 66 else "?"
        # β-barrel关键区域 144-148
        barrel_region = seq[143:149] if len(seq) > 148 else "?"
        wt_barrel = WT[143:149]
        barrel_changes = [(i+144, WT[i+143], seq[i+143]) for i in range(6) if i+143 < len(seq) and WT[i+143] != seq[i+143]]
        
        print(f"    发色团(65-67): {chromophore}")
        if barrel_changes:
            print(f"    β-barrel(144-149)变化: {', '.join(f'{a}{p}{b}' for p,a,b in barrel_changes)}")
        else:
            print(f"    β-barrel(144-149): 无变化 (与WT一致)")

print(f"\n{'='*70}")
print("  完成")
print(f"{'='*70}")
