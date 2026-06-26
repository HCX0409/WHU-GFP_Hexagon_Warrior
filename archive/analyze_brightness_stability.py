"""
分析 GFP_data 序列的亮度 vs 结构稳定性 (ddG) 关系
- 从 GFP_data 取各类型序列, 按亮度分 bin 采样
- 用 ESM3-ddG 预测相对 WT avGFP 的 ddG
- 可视化: 亮度-稳定性散点图, 相关性分析
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from src.config import WT_AVGFP, RANDOM_SEED, seed_everything
from src.phase2_generate import parse_mutations
from src.phase3_ddg_scoring import load_ddg_model, predict_ddg_single

GFP_TYPES = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
# 每种类型每个亮度 bin 采样数
SAMPLES_PER_BIN = 25
# 亮度 bins: (low, mid-low, mid, mid-high, high)
BRIGHTNESS_BINS = [
    (0.0, 2.0, "Dead/Low"),
    (2.0, 3.0, "Dim"),
    (3.0, 3.5, "Medium"),
    (3.5, 4.0, "Bright"),
    (4.0, 5.5, "Super Bright"),
]

BIN_COLORS = {
    "Dead/Low": "#d32f2f",
    "Dim": "#f57c00",
    "Medium": "#fbc02d",
    "Bright": "#388e3c",
    "Super Bright": "#1565c0",
}
TYPE_COLORS = {
    "avGFP": "#2196f3",
    "amacGFP": "#ff9800",
    "cgreGFP": "#4caf50",
    "ppluGFP": "#9c27b0",
}


def main():
    seed_everything()

    # ================= 1. 加载 GFP_data =================
    print("=" * 60)
    print("📂 加载 GFP_data ...")
    print("=" * 60)
    raw_df = pd.read_excel("data/raw/GFP_data.xlsx")
    gfp_col = raw_df.columns[1]
    print(f"  总记录: {len(raw_df)}")

    # ================= 2. 解析序列 + 按类型/亮度分层采样 =================
    print("\n" + "=" * 60)
    print("🧬 解析序列并分层采样 ...")
    print("=" * 60)

    all_data = []  # (sequence, brightness, gfp_type, aaMutations)
    for gfp_type in GFP_TYPES:
        type_df = raw_df[raw_df[gfp_col].str.contains(gfp_type, case=False, na=False)].copy()
        if len(type_df) == 0:
            continue
        print(f"\n  {gfp_type}: {len(type_df)} 条")

        rng = np.random.default_rng(RANDOM_SEED)
        sampled = []
        for (b_low, b_high, bin_name) in BRIGHTNESS_BINS:
            bin_df = type_df[(type_df['Brightness'] >= b_low) & (type_df['Brightness'] < b_high)]
            n = min(SAMPLES_PER_BIN, len(bin_df))
            if n > 0:
                idx = rng.choice(len(bin_df), size=n, replace=False)
                for i in idx:
                    row = bin_df.iloc[i]
                    seq = parse_mutations(str(row['aaMutations']), WT_AVGFP)
                    if seq and len(seq) == len(WT_AVGFP):
                        sampled.append({
                            'sequence': seq,
                            'brightness': row['Brightness'],
                            'gfp_type': gfp_type,
                            'aaMutations': str(row['aaMutations']),
                            'bin': bin_name,
                        })
                print(f"    [{bin_name}] {b_low:.1f}-{b_high:.1f}: {n} 条 (共 {len(bin_df)})")

        all_data.extend(sampled)

    sample_df = pd.DataFrame(all_data)
    print(f"\n  采样总计: {len(sample_df)} 条")

    # 去重 (同一序列只保留一条)
    sample_df = sample_df.drop_duplicates(subset=['sequence']).reset_index(drop=True)
    print(f"  去重后: {len(sample_df)} 条")

    # ================= 3. ESM3-ddG 预测 =================
    print("\n" + "=" * 60)
    print("🔬 加载 ESM3-ddG 并预测稳定性 ...")
    print("=" * 60)

    tokenizer, model, device = load_ddg_model()

    ddg_scores = []
    n_mutations = []
    errors = 0

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="ddG预测"):
        mut_seq = row['sequence']
        # 找到相对 WT avGFP 的所有突变位置
        positions = [i for i, (a, b) in enumerate(zip(WT_AVGFP, mut_seq)) if a != b]

        if len(positions) == 0:
            ddg_scores.append(0.0)
            n_mutations.append(0)
            continue

        if len(positions) > 30:
            # 太多突变 (可能是不同GFP类型的天然差异), 只采样前30个
            # 这样不会太慢, 也能得到大致的稳定性估计
            rng_pos = np.random.default_rng(RANDOM_SEED)
            positions = sorted(rng_pos.choice(positions, size=30, replace=False).tolist())

        try:
            total_ddg = predict_ddg_single(tokenizer, model, device, WT_AVGFP, mut_seq, positions)
            ddg_scores.append(total_ddg)
            n_mutations.append(len(positions))
        except Exception as e:
            ddg_scores.append(np.nan)
            n_mutations.append(len(positions))
            errors += 1
            if errors <= 3:
                print(f"  ⚠️ 预测失败 (seq {idx}): {e}")

    sample_df['ddG'] = ddg_scores
    sample_df['n_mutations_vs_wt'] = n_mutations
    valid = sample_df.dropna(subset=['ddG'])
    print(f"\n  预测完成: {len(valid)}/{len(sample_df)} 成功, {errors} 错误")

    # ================= 4. 分析 =================
    print("\n" + "=" * 60)
    print("📊 亮度 vs 稳定性 分析")
    print("=" * 60)

    # 4a. 整体相关性
    brightness = valid['brightness'].values
    ddg = valid['ddG'].values

    r_pearson, p_pearson = stats.pearsonr(brightness, ddg)
    r_spearman, p_spearman = stats.spearmanr(brightness, ddg)
    print(f"\n  整体 (N={len(valid)}):")
    print(f"    Pearson r = {r_pearson:.4f} (p={p_pearson:.2e})")
    print(f"    Spearman ρ = {r_spearman:.4f} (p={p_spearman:.2e})")
    print(f"    ddG范围: {ddg.min():.3f} ~ {ddg.max():.3f}")
    print(f"    亮度范围: {brightness.min():.3f} ~ {brightness.max():.3f}")

    # 4b. 分GFP类型
    print(f"\n  分GFP类型:")
    for gfp_type in GFP_TYPES:
        subset = valid[valid['gfp_type'] == gfp_type]
        if len(subset) < 5:
            continue
        b = subset['brightness'].values
        d = subset['ddG'].values
        r, p = stats.pearsonr(b, d)
        rs, ps = stats.spearmanr(b, d)
        avg_n_mut = subset['n_mutations_vs_wt'].mean()
        print(f"    {gfp_type} (N={len(subset)}, avg_muts={avg_n_mut:.1f}):")
        print(f"      Pearson r={r:.4f}, Spearman ρ={rs:.4f}")
        print(f"      ddG: {d.mean():.3f}±{d.std():.3f}")

    # 4c. 分亮度区间
    print(f"\n  分亮度区间 (全类型):")
    for (b_low, b_high, bin_name) in BRIGHTNESS_BINS:
        subset = valid[(valid['brightness'] >= b_low) & (valid['brightness'] < b_high)]
        if len(subset) < 3:
            continue
        d = subset['ddG'].values
        print(f"    [{bin_name}] {b_low:.1f}-{b_high:.1f}: N={len(subset)}, "
              f"ddG={d.mean():.3f}±{d.std():.3f}, "
              f"稳定化%: {(d < 0).sum() / len(d) * 100:.0f}%")

    # ================= 5. 可视化 =================
    print("\n" + "=" * 60)
    print("🎨 生成可视化 ...")
    print("=" * 60)

    # 5a. 总散点图 (按GFP类型着色)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    for gfp_type in GFP_TYPES:
        subset = valid[valid['gfp_type'] == gfp_type]
        if len(subset) == 0:
            continue
        ax.scatter(subset['brightness'], subset['ddG'],
                   c=TYPE_COLORS.get(gfp_type, 'gray'),
                   label=f"{gfp_type} (N={len(subset)})",
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
    ax.set_xlabel('Experimental Brightness (Log10)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted ddG (vs WT avGFP)', fontsize=12, fontweight='bold')
    ax.set_title(f'Brightness vs Stability (All Types)\nPearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 5b. 按亮度bin箱线图
    ax2 = axes[1]
    bin_data = []
    bin_labels = []
    for (b_low, b_high, bin_name) in BRIGHTNESS_BINS:
        subset = valid[(valid['brightness'] >= b_low) & (valid['brightness'] < b_high)]
        if len(subset) >= 3:
            bin_data.append(subset['ddG'].values)
            bin_labels.append(f"{bin_name}\n({b_low:.1f}-{b_high:.1f})")

    if bin_data:
        bp = ax2.boxplot(bin_data, labels=bin_labels, patch_artist=True,
                         medianprops=dict(color='red', linewidth=2))
        colors_list = [BIN_COLORS.get(BRIGHTNESS_BINS[i][2], 'gray')
                       for i in range(len(BRIGHTNESS_BINS))
                       if i < len(bin_data)]
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax2.set_xlabel('Brightness Bin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted ddG', fontsize=12, fontweight='bold')
    ax2.set_title('Stability Distribution by Brightness Range', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out1 = Path("brightness_vs_ddg_all.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"  💾 {out1}")
    plt.close()

    # 5c. 分GFP类型子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, gfp_type in enumerate(GFP_TYPES):
        ax = axes[idx // 2][idx % 2]
        subset = valid[valid['gfp_type'] == gfp_type]
        if len(subset) < 3:
            ax.set_title(f"{gfp_type} (N<3, skip)")
            continue

        b = subset['brightness'].values
        d = subset['ddG'].values
        r, p = stats.pearsonr(b, d)

        ax.scatter(b, d, c=TYPE_COLORS.get(gfp_type, 'gray'),
                   alpha=0.6, s=35, edgecolors='white', linewidth=0.3)

        # 回归线
        z = np.polyfit(b, d, 1)
        x_line = np.linspace(b.min(), b.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Brightness (Log10)', fontsize=11)
        ax.set_ylabel('ddG', fontsize=11)
        ax.set_title(f"{gfp_type} (N={len(subset)}, r={r:.3f})", fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = Path("brightness_vs_ddg_by_type.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"  💾 {out2}")
    plt.close()

    # ================= 6. 保存数据 =================
    out_csv = Path("brightness_ddg_analysis.csv")
    valid.to_csv(out_csv, index=False)
    print(f"\n  💾 原始数据: {out_csv}")

    print("\n" + "=" * 60)
    print("✅ 分析完成!")
    print("=" * 60)
    print(f"  核心问题: 高亮度序列是否更稳定 (ddG更负)?")
    print(f"  Pearson r = {r_pearson:.4f} → ", end="")
    if abs(r_pearson) < 0.1:
        print("几乎无相关性")
    elif r_pearson < -0.3:
        print("负相关 (亮度越高, ddG越低/越稳定)")
    elif r_pearson > 0.3:
        print("正相关 (亮度越高, ddG越高/越不稳定)")
    else:
        print("弱相关")


if __name__ == "__main__":
    main()
