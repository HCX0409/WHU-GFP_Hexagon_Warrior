"""
V6 训练数据准备: 高亮度聚焦采样
策略:
  - cgreGFP/ppluGFP 尽可能多 (高端亮度专家)
  - avGFP/amacGFP 补线性 (低端守卫知识)
  - 分布从低到高递增, 高端集中 50%+ 数据

产出: data/processed/v6_training_data.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.config import (
    PROCESSED_DIR, RAW_DIR, WT_AVGFP, RANDOM_SEED, AUTHOR, DATE_STR
)
from src.phase1_data import parse_mutations

# ======================== 采样配额表 ========================
# 格式: (brightness_low, brightness_high, bin_name,
#        avGFP_cap, amacGFP_cap, cgreGFP_cap, ppluGFP_cap)
SAMPLING_PLAN = [
    # 低亮度: 少量锚点
    (0.0, 1.5, "Dead",       200,   0,     0,     0),
    (1.5, 2.5, "VeryLow",    300,   0,     0,     0),
    # 中低: 线性过渡
    (2.5, 3.0, "Low",        500,   1000,  1000,  800),
    (3.0, 3.5, "Medium",     1000,  1200,  None,  None),  # None = 全取
    # 中高: 接近目标
    (3.5, 4.0, "Bright",     1500,  2000,  None,  3000),
    # 高亮: 核心区域
    (4.0, 4.3, "High",       None,  None,  None,  5000),
    # 超亮: 最有价值
    (4.3, 5.5, "Ultra",      0,     0,     None,  None),
]

GFP_TYPES = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']

WT_BRIGHTNESS = {
    'avGFP': 3.7192,
    'amacGFP': 3.9707,
    'cgreGFP': 4.4969,
    'ppluGFP': 4.2258
}


def prepare_v6_data(force: bool = False):
    """构建V6训练数据集"""
    out_path = PROCESSED_DIR / "v6_training_data.csv"
    if out_path.exists() and not force:
        print(f"⚠️ 已存在: {out_path}")
        print("   强制重跑: python main.py prepare_v6 --force")
        return out_path

    print("=" * 60)
    print("📊 V6 训练数据准备 (高亮度聚焦采样)")
    print("=" * 60)

    # 1. 加载原始数据
    raw_df = pd.read_excel(RAW_DIR / "GFP_data.xlsx")
    gfp_col = raw_df.columns[1]
    print(f"  原始数据: {len(raw_df)} 条")

    # 2. 预解析所有序列
    print("\n-> 解析序列...")
    parsed = {}  # gfp_type -> list of (seq, brightness, aaMutations)
    for gfp_type in GFP_TYPES:
        type_df = raw_df[raw_df[gfp_col].str.contains(
            gfp_type, case=False, na=False)].copy()
        print(f"  {gfp_type}: {len(type_df)} 条")

        items = []
        for _, row in tqdm(type_df.iterrows(), total=len(type_df),
                           desc=f"  解析{gfp_type}", leave=False):
            seq = parse_mutations(str(row['aaMutations']), WT_AVGFP)
            if seq and len(seq) == len(WT_AVGFP):
                items.append({
                    'sequence': seq,
                    'brightness': row['Brightness'],
                    'aaMutations': str(row['aaMutations']),
                    'gfp_type': gfp_type,
                })
        print(f"    有效序列: {len(items)} 条")
        parsed[gfp_type] = items

    # 3. 按采样计划执行
    print("\n" + "=" * 60)
    print("🎯 分层采样")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)
    all_sampled = []
    type_idx = {t: i for i, t in enumerate(GFP_TYPES)}

    header = f"{'Bin':<12} {'avGFP':>8} {'amacGFP':>8} {'cgreGFP':>8} {'ppluGFP':>8} {'Total':>8}"
    print(header)
    print("-" * 56)

    for (b_lo, b_hi, bin_name, *caps) in SAMPLING_PLAN:
        row_line = f"{bin_name:<12}"
        bin_total = 0

        for i, gfp_type in enumerate(GFP_TYPES):
            cap = caps[i]
            if cap == 0:
                row_line += f"{'--':>8}"
                continue

            # 从该类型的有效序列中筛选
            items = [it for it in parsed[gfp_type]
                     if b_lo <= it['brightness'] < b_hi]
            n_avail = len(items)

            if n_avail == 0:
                row_line += f"{'0':>8}"
                continue

            if cap is None:
                # 全取
                selected = items
            else:
                n = min(cap, n_avail)
                idx = rng.choice(n_avail, size=n, replace=False)
                selected = [items[j] for j in idx]

            # 去重 (基于序列)
            seen = set()
            deduped = []
            for it in selected:
                key = (it['sequence'], it['gfp_type'])
                if key not in seen:
                    seen.add(key)
                    deduped.append(it)
            selected = deduped

            all_sampled.extend(selected)
            row_line += f"{len(selected):>8}"
            bin_total += len(selected)

        row_line += f"{bin_total:>8}"
        print(row_line)

    # 4. 构建 DataFrame (与V5一致: sequence+brightness+gfp_type)
    df = pd.DataFrame(all_sampled)
    df = df[['sequence', 'brightness', 'gfp_type']]  # 这三列

    # 打乱顺序
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # 5. 统计
    print(f"\n{'='*60}")
    print("📈 V6 训练集统计")
    print(f"{'='*60}")
    print(f"  总量: {len(df)} 条")
    
    print(f"\n  分GFP类型:")
    for gt in GFP_TYPES:
        sub = df[df['gfp_type'] == gt]
        if len(sub) == 0:
            continue
        b = sub['brightness']
        print(f"    {gt}: N={len(sub):>6}, "
              f"brightness={b.min():.2f}~{b.max():.2f}, "
              f"mean={b.mean():.3f}±{b.std():.3f}")

    print(f"\n  分亮度区间:")
    for (b_lo, b_hi, bin_name, *_) in SAMPLING_PLAN:
        sub = df[(df['brightness'] >= b_lo) & (df['brightness'] < b_hi)]
        if len(sub) > 0:
            print(f"    [{bin_name}] {b_lo:.1f}-{b_hi:.1f}: "
                  f"N={len(sub):>6} ({len(sub)/len(df)*100:.1f}%)")

    # 6. 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n💾 存档: {out_path}")
    print(f"💡 V6训练: python main.py train_v6 --force")
    return out_path


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    prepare_v6_data(force=force)
