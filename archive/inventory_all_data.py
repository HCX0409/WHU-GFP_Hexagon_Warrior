"""全面盘点所有已生成的序列数据"""
import pandas as pd
from pathlib import Path

files = [
    ('phase2_bright_mut_5000_HCX_260611.csv', 'Phase2原始5000条(6/11)'),
    ('phase3_ddg_scored_HCX_260612.csv', 'Phase3 ddG打分(6/12)'),
    ('phase4_final_top20_HCX_260613.csv', 'Phase4最终Top20(6/13)'),
    ('phase2_crossval_HCX_260620.csv', 'Phase2交叉验证(6/20)'),
    ('phase2_full_scored_HCX_260623.csv', 'Phase2.5完整打分(6/23)'),
    ('phase2_hierarchical_HCX_260623.csv', 'Phase2.5共识结果(6/23)'),
]

for fname, label in files:
    p = Path('generated_seqs') / fname
    if not p.exists():
        print(f"\n{label}: 不存在")
        continue
    df = pd.read_csv(p)
    print(f"\n{'='*60}")
    print(f"{label}: {len(df)}条, {list(df.columns)[:8]}{'...' if len(df.columns)>8 else ''}")
    
    # 检查是否有gfp_type列
    if 'gfp_type' in df.columns:
        print(f"  GFP类型分布: {dict(df['gfp_type'].value_counts())}")
    elif 'GFP_type' in df.columns:
        print(f"  GFP类型分布: {dict(df['GFP_type'].value_counts())}")
    
    # 检查是否有预测分数列
    for col in ['pred_v7', 'pred_v9_abs', 'pred_v5_abs', 'weighted_avg', 'votes',
                'pred_v9', 'pred_v5', 'ddg_mean', 'brightness_pred', 'consensus_score']:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  {col}: mean={vals.mean():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")
    
    # 显示前3条序列的关键信息
    if 'sequence' in df.columns:
        print(f"  序列长度: {df['sequence'].str.len().unique()}")
        # 检查突变数
        if 'n_mutations' in df.columns:
            print(f"  突变数分布: {dict(df['n_mutations'].value_counts().sort_index())}")
