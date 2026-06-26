"""
Phase 4: 最终筛选 (ddG主排 + Exclusion过滤 + 多样性配额)
  - 读取 Phase 3 产出 (含 pred_brightness + total_ddg)
  - Exclusion List 过滤
  - ddG 为主排名指标 (亮度作为辅助)
  - 每种GFP类型至少3条配额
  - 输出 Top 20

输入: Phase 3 CSV
产出: generated_seqs/phase4_final_top20_{AUTHOR}_{DATE}.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch

from src.config import (
    GEN_SEQS_DIR, RAW_DIR, AUTHOR, DATE_STR, RANDOM_SEED, seed_everything
)
from src.phase2_generate import GFP_TYPES


# ======================== 常量 ========================
TOP_N_FINAL = 20
MIN_PER_TYPE = 3    # 每种GFP类型至少3条
# 综合评分: 0.3*亮度zscore + 0.7*(-ddG zscore)
W_BRIGHTNESS = 0.3
W_STABILITY = 0.7


def zscore(series):
    """标准化到z-score"""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.mean()) / std


def score_and_filter(force: bool = False):
    """
    Phase 4: 最终筛选 → Top 20
    """
    seed_everything()

    print("\n" + "=" * 60)
    print(f"🏆 [Phase 4] 最终筛选 (ddG主排 + 多样性配额)")
    print("=" * 60)

    # 1. 读取 Phase 3 产出
    phase3_files = sorted(GEN_SEQS_DIR.glob("phase3_ddg_scored_HCX_*.csv"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    if not phase3_files:
        raise FileNotFoundError("❌ 未找到 Phase 3 产出! 请先运行: python main.py score_ddg")

    phase3_path = phase3_files[0]
    print(f"-> 读取 Phase 3: {phase3_path.name}")
    df = pd.read_csv(phase3_path)
    print(f"   总序列: {len(df)}")

    # 只保留突变变体 (不要父本)
    df = df[df['source'] == 'bright_mut'].copy()
    print(f"   突变变体: {len(df)} 条")

    # 2. Exclusion List 过滤
    excl_path = RAW_DIR / "Exclusion_List.csv"
    if excl_path.exists():
        excl_df = pd.read_csv(excl_path)
        exclusion_set = set(excl_df['Sequence'].str.strip().tolist())
        n_excl = df['sequence'].isin(exclusion_set).sum()
        df = df[~df['sequence'].isin(exclusion_set)]
        print(f"   Exclusion过滤: 排除 {n_excl} 条, 剩余 {len(df)} 条")
    else:
        print("⚠️ Exclusion List 不存在")

    # 3. 去重
    df = df.drop_duplicates(subset=['sequence'], keep='first')
    print(f"   去重后: {len(df)} 条")

    # 4. 综合评分: z-score加权
    # 亮度: 越高越好 → z-score
    # ddG: 越低越稳定 → -ddG越高越好 → z-score
    df['brightness_z'] = zscore(df['pred_brightness'])
    df['stability_z'] = zscore(-df['total_ddg'])  # 取反: ddG负=稳定 → 正值好
    df['combined_score'] = W_BRIGHTNESS * df['brightness_z'] + W_STABILITY * df['stability_z']

    print(f"\n{'='*40}")
    print(f"📊 评分统计")
    print(f"{'='*40}")
    print(f"  亮度: {df['pred_brightness'].min():.3f} ~ {df['pred_brightness'].max():.3f} (均值 {df['pred_brightness'].mean():.3f})")
    print(f"  ddG:  {df['total_ddg'].min():.3f} ~ {df['total_ddg'].max():.3f} (均值 {df['total_ddg'].mean():.3f})")
    print(f"  综合分: {df['combined_score'].min():.3f} ~ {df['combined_score'].max():.3f}")

    # 5. 多样性配额选择
    print(f"\n{'='*40}")
    print(f"🎯 多样性配额选择 (每种GFP≥{MIN_PER_TYPE}, 总{TOP_N_FINAL})")
    print(f"{'='*40}")

    selected_ids = set()

    # 先每种类型选 MIN_PER_TYPE 条
    for gfp_type in GFP_TYPES:
        type_df = df[df['gfp_type'] == gfp_type].nlargest(MIN_PER_TYPE, 'combined_score')
        selected_ids.update(type_df.index.tolist())
        n = len(type_df)
        if n > 0:
            print(f"  {gfp_type}: 保底 {n} 条 (最佳综合={type_df['combined_score'].max():.3f})")

    # 剩余名额按综合分填满
    remaining_slots = TOP_N_FINAL - len(selected_ids)
    if remaining_slots > 0:
        remaining = df[~df.index.isin(selected_ids)].nlargest(remaining_slots, 'combined_score')
        selected_ids.update(remaining.index.tolist())

    final = df[df.index.isin(selected_ids)].sort_values(
        'combined_score', ascending=False
    ).reset_index(drop=True)

    final['seq_id'] = [f"final_{i:03d}" for i in range(len(final))]

    # 6. 输出统计
    print(f"\n{'='*60}")
    print(f"🏆 最终 Top {len(final)}:")
    print(f"{'='*60}")

    for gfp_type in GFP_TYPES:
        type_final = final[final['gfp_type'] == gfp_type]
        if len(type_final) > 0:
            print(f"   {gfp_type}: {len(type_final)} 条 "
                  f"(亮度 {type_final['pred_brightness'].min():.3f}~{type_final['pred_brightness'].max():.3f}, "
                  f"ddG {type_final['total_ddg'].min():.3f}~{type_final['total_ddg'].max():.3f})")
        else:
            print(f"   {gfp_type}: 0 条")

    print(f"\n   综合分范围: {final['combined_score'].min():.4f} ~ {final['combined_score'].max():.4f}")
    print(f"   亮度范围:   {final['pred_brightness'].min():.4f} ~ {final['pred_brightness'].max():.4f}")
    print(f"   ddG范围:    {final['total_ddg'].min():.4f} ~ {final['total_ddg'].max():.4f}")

    # Top 5 详情
    print(f"\n📝 Top 5 详情:")
    for i, row in final.head(5).iterrows():
        print(f"  #{i+1}: 综合={row['combined_score']:.4f}, "
              f"亮度={row['pred_brightness']:.4f}, "
              f"ddG={row['total_ddg']:+.4f}, "
              f"突变={row.get('num_mutations_from_parent', '?')}, "
              f"类型={row['gfp_type']}")
        print(f"       序列: {row['sequence'][:70]}...")

    # 存档
    out_path = GEN_SEQS_DIR / f"phase4_final_top{TOP_N_FINAL}_{AUTHOR}_{DATE_STR}.csv"
    final.to_csv(out_path, index=False)
    print(f"\n💾 存档: {out_path}")
    print(f"💡 这 {len(final)} 条序列可直接提交或送 AlphaFold3 验证 (Phase 5)")

    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: 最终筛选")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    score_and_filter(force=args.force)
