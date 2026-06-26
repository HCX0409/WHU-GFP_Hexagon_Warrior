"""快速分析全量数据分布, 设计72k筛选策略"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.phase1_data import parse_mutations

WT_BRIGHTNESS = {
    'avGFP': 3.7192, 'amacGFP': 3.9707,
    'cgreGFP': 4.4969, 'ppluGFP': 4.2258
}

# 从V5用的WT序列
from src.config import LOCAL_ESM2_DIR
from transformers import AutoTokenizer

raw_df = pd.read_excel("data/raw/GFP_data.xlsx")
gfp_col = raw_df.columns[1]
print(f"原始数据: {len(raw_df)} 条")
print(f"列名: {list(raw_df.columns)}")
print(f"\n各类型分布:")
print(raw_df[gfp_col].value_counts())

# 解析序列并计算delta
all_data = []
for idx, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="解析"):
    gfp_type = row[gfp_col]
    mut_str = row['aaMutations']
    brightness = row['Brightness']
    
    seq = parse_mutations(mut_str, gfp_type)
    if seq is None:
        continue
    
    wt_bright = WT_BRIGHTNESS.get(gfp_type, 3.7192)
    delta = brightness - wt_bright
    
    # 突变数量
    n_mutations = 0
    if pd.notna(mut_str) and mut_str != 'WT' and mut_str != 'wt':
        n_mutations = len(str(mut_str).split(','))
    
    all_data.append({
        'gfp_type': gfp_type,
        'brightness': brightness,
        'delta': delta,
        'n_mutations': n_mutations,
        'seq_len': len(seq)
    })

df = pd.DataFrame(all_data)
print(f"\n解析后: {len(df)} 条")

# 去重
df_unique = df.drop_duplicates(subset=['gfp_type', 'brightness', 'n_mutations'])
# 用完整序列去重更准确, 但这里用近似
print(f"去重后(近似): {len(df_unique)} 条")

print(f"\n{'='*60}")
print("各类型分布及delta统计:")
print(f"{'='*60}")
for gt in ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']:
    sub = df[df['gfp_type'] == gt]
    print(f"\n{gt} (N={len(sub)}, {len(sub)/len(df)*100:.1f}%):")
    print(f"  亮度: {sub['brightness'].min():.3f} ~ {sub['brightness'].max():.3f}, "
          f"mean={sub['brightness'].mean():.3f}")
    print(f"  delta: {sub['delta'].min():.3f} ~ {sub['delta'].max():.3f}, "
          f"mean={sub['delta'].mean():.3f}, std={sub['delta'].std():.3f}")
    print(f"  突变数: {sub['n_mutations'].mean():.1f} 平均")
    
    # delta分布
    for threshold in [0, 0.3, 0.5, -0.3, -0.5]:
        if threshold >= 0:
            n = (sub['delta'] > threshold).sum()
            print(f"    delta>{threshold}: {n} ({n/len(sub)*100:.1f}%)")
        else:
            n = (sub['delta'] < threshold).sum()
            print(f"    delta<{threshold}: {n} ({n/len(sub)*100:.1f}%)")

# 72k筛选策略分析
print(f"\n{'='*60}")
print("72k筛选策略模拟:")
print(f"{'='*60}")

# 策略: 保留大delta + 高亮度, 去掉近WT
# 分层: |delta|>0.5全保留, |delta|>0.3优先, 其余按比例采样
df_sorted = df.sort_values('delta', ascending=False)

# 优先级:
# Tier 1: |delta| > 0.5 (大幅变化, 必须保留)
# Tier 2: brightness >= WT + 0.3 (有意义的亮度提升)
# Tier 3: |delta| > 0.2 (中等变化)
# Tier 4: 其余

tier1 = df[abs(df['delta']) > 0.5]
tier2 = df[(df['delta'] > 0.3) & (abs(df['delta']) <= 0.5)]
tier3 = df[(abs(df['delta']) > 0.2) & (abs(df['delta']) <= 0.3)]
tier4 = df[abs(df['delta']) <= 0.2]

print(f"  Tier 1 (|delta|>0.5): {len(tier1)}")
print(f"  Tier 2 (0.3<delta<=0.5): {len(tier2)}")
print(f"  Tier 3 (0.2<|delta|<=0.3): {len(tier3)}")
print(f"  Tier 4 (|delta|<=0.2): {len(tier4)}")
print(f"  总计: {len(tier1)+len(tier2)+len(tier3)+len(tier4)}")

# 目标72k
target = 72000
if len(tier1) + len(tier2) + len(tier3) >= target:
    print(f"\n  仅Tier1+2+3就有 {len(tier1)+len(tier2)+len(tier3)} 条, 足够72k")
    print(f"  可以从Tier3中随机采样来凑到72k")
elif len(tier1) + len(tier2) >= target:
    print(f"\n  Tier1+2有 {len(tier1)+len(tier2)} 条")
else:
    print(f"\n  Tier1+2+3+4共 {len(tier1)+len(tier2)+len(tier3)+len(tier4)} 条")
    remaining = target - len(tier1) - len(tier2) - len(tier3)
    print(f"  需从Tier4中采样 {remaining} 条")
