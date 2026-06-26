"""分析3轮迭代的V7通过数据"""
import pandas as pd

all_dfs = []
for i in [1, 2, 3]:
    df = pd.read_csv(f'generated_seqs/phase2_iter{i}_HCX_260620.csv')
    df['iter'] = i
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
print(f'3轮合计: {len(combined)} 条')
n_uniq = combined.drop_duplicates(subset='sequence').shape[0]
print(f'去重后: {n_uniq} 条')
print()

# 按类型统计
for gt in ['cgreGFP', 'ppluGFP', 'amacGFP', 'avGFP']:
    sub = combined[combined.gfp_type == gt]
    sub_u = sub.drop_duplicates(subset='sequence')
    print(f'{gt}: 原始{len(sub)}条, 去重{len(sub_u)}条')
    if len(sub_u) > 0:
        top5 = sub_u.sort_values('pred_v7', ascending=False).head(5)
        for _, r in top5.iterrows():
            print(f'  V7={r.pred_v7:.3f} mut={r.n_mutations} parent_bright={r.parent_brightness:.3f} rank={r.parent_rank}')
print()

# 检查每轮最低V7 (判断是否截断在500条)
print('每轮最低V7:')
for i in [1, 2, 3]:
    df = all_dfs[i-1]
    for gt in ['cgreGFP', 'ppluGFP']:
        sub = df[df.gfp_type == gt]
        if len(sub) > 0:
            print(f'  iter{i} {gt}: min={sub.pred_v7.min():.4f} max={sub.pred_v7.max():.4f} count={len(sub)}')
        else:
            print(f'  iter{i} {gt}: 0条')
