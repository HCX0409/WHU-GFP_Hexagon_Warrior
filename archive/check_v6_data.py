"""快速分析V6训练数据的delta分布"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.config import PROCESSED_DIR

df = pd.read_csv(PROCESSED_DIR / "v6_training_data.csv")

print("=" * 70)
print("V6 训练数据详细分析")
print("=" * 70)
print(f"总量: {len(df)} 条\n")

# 1. 总体delta分布
print("【1】总体 delta 分布")
print(f"  delta 范围: {df['brightness_delta'].min():.4f} ~ {df['brightness_delta'].max():.4f}")
print(f"  delta 均值: {df['brightness_delta'].mean():.4f}")
print(f"  delta 标准差: {df['brightness_delta'].std():.4f}")
print(f"  delta 中位数: {df['brightness_delta'].median():.4f}")

# 2. 分GFP类型
GFP_TYPES = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
WT = {'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258}

print(f"\n{'='*70}")
print("【2】分 GFP 类型")
print(f"{'类型':<10} {'N':>6} {'亮度范围':>16} {'delta均值':>10} {'delta std':>10} {'delta范围':>18}")
print("-" * 70)
for gt in GFP_TYPES:
    sub = df[df['gfp_type'] == gt]
    if len(sub) == 0:
        continue
    b = sub['brightness']
    d = sub['brightness_delta']
    print(f"{gt:<10} {len(sub):>6} {b.min():.2f}~{b.max():.2f}    "
          f"{d.mean():>+8.3f}  {d.std():>8.3f}  {d.min():+.2f}~{d.max():+.2f}")

# 3. 关键问题：delta>0 有多少？
print(f"\n{'='*70}")
print("【3】delta > 0 的样本 (比WT更亮的序列)")
print(f"{'类型':<10} {'总量':>6} {'delta>0':>8} {'占比':>8} {'这些delta均值':>12}")
print("-" * 50)
for gt in GFP_TYPES:
    sub = df[df['gfp_type'] == gt]
    pos = sub[sub['brightness_delta'] > 0]
    pct = len(pos) / len(sub) * 100 if len(sub) > 0 else 0
    m = pos['brightness_delta'].mean() if len(pos) > 0 else float('nan')
    print(f"{gt:<10} {len(sub):>6} {len(pos):>8} {pct:>7.1f}% {m:>+11.3f}")

total_pos = (df['brightness_delta'] > 0).sum()
print(f"\n  总计 delta>0: {total_pos}/{len(df)} ({total_pos/len(df)*100:.1f}%)")

# 4. delta的分位数
print(f"\n{'='*70}")
print("【4】delta 分位数分布")
for p in [5, 10, 25, 50, 75, 90, 95]:
    v = np.percentile(df['brightness_delta'], p)
    print(f"  P{p:>2}: {v:>+.4f}")

# 5. 绝对亮度的分位数
print(f"\n{'='*70}")
print("【5】绝对亮度 分位数分布")
for p in [5, 10, 25, 50, 75, 90, 95]:
    v = np.percentile(df['brightness'], p)
    print(f"  P{p:>2}: {v:.4f}")

# 6. 如果模型只预测平均值，R²会是多少？
print(f"\n{'='*70}")
print("【6】基线分析: 如果模型预测全部均值")
mean_delta = df['brightness_delta'].mean()
ss_res = ((df['brightness_delta'] - mean_delta) ** 2).sum()
ss_tot = ss_res  # 预测均值时 R²=0
mae_baseline = (df['brightness_delta'] - mean_delta).abs().mean()
print(f"  全部均值预测: delta={mean_delta:+.4f}")
print(f"  基线MAE: {mae_baseline:.4f}")

# 分类型预测均值的MAE
total_err = []
for gt in GFP_TYPES:
    sub = df[df['gfp_type'] == gt]['brightness_delta']
    if len(sub) == 0:
        continue
    type_mean = sub.mean()
    err = (sub - type_mean).abs().mean()
    total_err.extend((sub - type_mean).abs().tolist())
    print(f"  {gt} 均值预测: delta={type_mean:+.4f}, MAE={err:.4f}")

overall_type_mae = np.mean(total_err)
print(f"  分类型均值预测的整体MAE: {overall_type_mae:.4f}")
print(f"  (这代表: 即使模型什么都不学, 只记住每种GFP的平均delta, MAE就是{overall_type_mae:.4f})")
