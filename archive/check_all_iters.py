"""检查所有已保存的迭代数据"""
import pandas as pd
from pathlib import Path

files = [
    ('generated_seqs/phase2_iter1_HCX_260620.csv', 'iter1(旧阈值)'),
    ('generated_seqs/phase2_iter2_HCX_260620.csv', 'iter2(旧阈值)'),
    ('generated_seqs/phase2_iter3_HCX_260620.csv', 'iter3(旧阈值)'),
    ('generated_seqs/phase2_iter4_HCX_260621.csv', 'iter4(新阈值)'),
]

all_dfs = []
for path, label in files:
    p = Path(path)
    if not p.exists():
        print(f"{label}: 文件不存在")
        continue
    df = pd.read_csv(p)
    df['source'] = label
    all_dfs.append(df)
    print(f"\n{label}: {len(df)}条")
    for gt in ['cgreGFP', 'ppluGFP', 'amacGFP', 'avGFP']:
        sub = df[df.gfp_type == gt]
        if len(sub) > 0:
            print(f"  {gt}: {len(sub)}条, V7=[{sub.pred_v7.min():.3f}, {sub.pred_v7.max():.3f}]")
        else:
            print(f"  {gt}: 0条")

# 合并所有
if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_u = combined.drop_duplicates(subset='sequence', keep='first')
    print(f"\n{'='*50}")
    print(f"全部合并: {len(combined)}条, 去重: {len(combined_u)}条")
    for gt in ['cgreGFP', 'ppluGFP', 'amacGFP', 'avGFP']:
        sub = combined_u[combined_u.gfp_type == gt]
        print(f"  {gt}: {len(sub)}条")
        if len(sub) > 0:
            top3 = sub.sort_values('pred_v7', ascending=False).head(3)
            for _, r in top3.iterrows():
                print(f"    V7={r.pred_v7:.3f} mut={r.n_mutations} parent_b={r.parent_brightness:.3f}")
