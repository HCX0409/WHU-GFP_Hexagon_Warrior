"""
Phase 1.7: 从141k全量数据中筛选72k优质数据
筛选策略:
  - Tier 1: |delta| > 0.5 (大幅亮度变化, 全部保留)
  - Tier 2: delta > 0.3 (有意义的亮度提升, 全部保留)  
  - Tier 3: 多突变序列 (>=3个突变, 全部保留)
  - Tier 4: 按亮度均匀分层采样, 确保全范围覆盖
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.phase1_data import parse_mutations
from src.config import PROCESSED_DIR, RANDOM_SEED, seed_everything

# WT亮度基准
WT_BRIGHTNESS = {
    'avGFP': 3.7192, 'amacGFP': 3.9707,
    'cgreGFP': 4.4969, 'ppluGFP': 4.2258
}

# avGFP WT序列 (用于parse_mutations)
WT_AVGFP = ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"
             "LSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGI"
             "DFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGP"
             "VLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK")

TARGET_COUNT = 72000
OUTPUT_PATH = PROCESSED_DIR / "v7_72k_quality_data.csv"


def prepare_72k_data(force=False):
    seed_everything(RANDOM_SEED)
    
    if OUTPUT_PATH.exists() and not force:
        print(f"⚠️ 已存在: {OUTPUT_PATH}")
        print(f"   如需重新生成请加 --force")
        return
    
    # 1. 读取全量数据 (brightness sheet)
    print("=" * 60)
    print("Step 1: 读取全量数据")
    raw_df = pd.read_excel("data/raw/GFP_data.xlsx", sheet_name="brightness")
    print(f"  原始数据: {len(raw_df)} 条")
    print(f"  列名: {list(raw_df.columns)}")
    gfp_col = 'GFP type'
    print(f"  各类型: {raw_df[gfp_col].value_counts().to_dict()}")
    
    # 2. 解析全长序列
    print(f"\n{'='*60}")
    print("Step 2: 解析突变序列")
    all_records = []
    fail_count = 0
    
    for idx, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="解析中"):
        gfp_type = row[gfp_col]
        mut_str = row['aaMutations']
        brightness = row['Brightness']
        
        # 所有突变都应用到avGFP WT (因为序列相似度极高)
        seq = parse_mutations(mut_str, WT_AVGFP)
        if seq is None:
            fail_count += 1
            continue
        
        wt_bright = WT_BRIGHTNESS.get(gfp_type, 3.7192)
        delta = brightness - wt_bright
        
        # 突变数量 (用冒号分隔)
        n_mut = 0
        if pd.notna(mut_str) and str(mut_str).upper() not in ('WT', ''):
            n_mut = len(str(mut_str).split(':'))
        
        all_records.append({
            'sequence': seq,
            'brightness': brightness,
            'gfp_type': gfp_type,
            'delta': delta,
            'n_mutations': n_mut
        })
    
    df = pd.DataFrame(all_records)
    print(f"  解析成功: {len(df)}, 失败: {fail_count}")
    
    # 3. 去重 (同一序列+同一GFP类型)
    print(f"\n{'='*60}")
    print("Step 3: 去重")
    before = len(df)
    df = df.drop_duplicates(subset=['sequence', 'gfp_type'])
    print(f"  {before} → {len(df)} 条 (去重 {before - len(df)} 条)")
    
    # 4. 质量清洗
    print(f"\n{'='*60}")
    print("Step 4: 质量清洗")
    
    # 剔除死值 (出现>50次的亮度值)
    brightness_counts = df['brightness'].value_counts()
    dead_values = brightness_counts[brightness_counts > 50].index
    if len(dead_values) > 0:
        df = df[~df['brightness'].isin(dead_values)]
        print(f"  剔除死值: {len(dead_values)} 个 ({dead_values.tolist()})")
    
    # 剔除负数亮度
    df = df[df['brightness'] >= 0]
    print(f"  清洗后: {len(df)} 条")
    
    # 5. 分类型统计
    print(f"\n{'='*60}")
    print("Step 5: 各类型分布")
    for gt in ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']:
        sub = df[df['gfp_type'] == gt]
        print(f"  {gt}: N={len(sub)} ({len(sub)/len(df)*100:.1f}%), "
              f"brightness=[{sub['brightness'].min():.2f}, {sub['brightness'].max():.2f}], "
              f"delta=[{sub['delta'].min():.3f}, {sub['delta'].max():.3f}]")
    
    # 6. 分层筛选
    print(f"\n{'='*60}")
    print(f"Step 6: 筛选 {TARGET_COUNT} 条优质数据")
    
    # Tier定义
    df['tier'] = 4  # 默认Tier 4
    
    # Tier 1: |delta| > 0.5 (大幅变化)
    mask_t1 = abs(df['delta']) > 0.5
    df.loc[mask_t1, 'tier'] = 1
    
    # Tier 2: delta > 0.3 (亮度提升)
    mask_t2 = (df['delta'] > 0.3) & (df['tier'] > 1)
    df.loc[mask_t2, 'tier'] = 2
    
    # Tier 3: 多突变 (>=3)
    mask_t3 = (df['n_mutations'] >= 3) & (df['tier'] > 2)
    df.loc[mask_t3, 'tier'] = 3
    
    for t in range(1, 5):
        n = (df['tier'] == t).sum()
        print(f"  Tier {t}: {n} 条")
    
    # 按优先级采样
    tier1 = df[df['tier'] == 1]
    tier2 = df[df['tier'] == 2]
    tier3 = df[df['tier'] == 3]
    tier4 = df[df['tier'] == 4]
    
    selected = []
    remaining = TARGET_COUNT
    
    # Tier 1-3 全部保留
    for tier_name, tier_df in [("Tier 1", tier1), ("Tier 2", tier2), ("Tier 3", tier3)]:
        if len(tier_df) <= remaining:
            selected.append(tier_df)
            remaining -= len(tier_df)
            print(f"  {tier_name}: 全部保留 {len(tier_df)} 条, 还需 {remaining}")
        else:
            sampled = tier_df.sample(n=remaining, random_state=RANDOM_SEED)
            selected.append(sampled)
            remaining = 0
            print(f"  {tier_name}: 采样 {len(sampled)} 条, 已满")
            break
    
    # 如果还不够, 从Tier 4分层采样
    if remaining > 0:
        print(f"  从Tier 4采样 {remaining} 条 (亮度分层)")
        # 按亮度10分位数分层, 保证全范围覆盖
        tier4 = tier4.copy()
        tier4['brightness_bin'] = pd.qcut(tier4['brightness'], q=10, duplicates='drop')
        
        # 每层采样
        per_bin = remaining // tier4['brightness_bin'].nunique()
        extra = remaining - per_bin * tier4['brightness_bin'].nunique()
        
        tier4_sampled = []
        for bin_val, group in tier4.groupby('brightness_bin'):
            n_sample = min(len(group), per_bin)
            tier4_sampled.append(group.sample(n=n_sample, random_state=RANDOM_SEED))
        
        tier4_final = pd.concat(tier4_sampled)
        
        # 如果还不够, 从剩余Tier 4中补充
        if len(tier4_final) < remaining:
            used_idx = tier4_final.index
            leftover = tier4[~tier4.index.isin(used_idx)]
            need_more = remaining - len(tier4_final)
            if len(leftover) >= need_more:
                tier4_final = pd.concat([tier4_final, leftover.sample(n=need_more, random_state=RANDOM_SEED)])
            else:
                tier4_final = pd.concat([tier4_final, leftover])
        
        tier4_final = tier4_final.head(remaining)
        selected.append(tier4_final)
        print(f"  Tier 4: 采样 {len(tier4_final)} 条")
    
    # 合并
    result = pd.concat(selected)
    result = result.drop(columns=['tier', 'brightness_bin'], errors='ignore')
    
    print(f"\n{'='*60}")
    print(f"最终数据集: {len(result)} 条")
    print(f"  各类型: {result['gfp_type'].value_counts().to_dict()}")
    print(f"  亮度范围: [{result['brightness'].min():.3f}, {result['brightness'].max():.3f}]")
    print(f"  delta范围: [{result['delta'].min():.3f}, {result['delta'].max():.3f}]")
    print(f"  突变数均值: {result['n_mutations'].mean():.1f}")
    
    # 保存
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 保存至: {OUTPUT_PATH}")
    print(f"列: {list(result.columns)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_72k_data(force=args.force)
