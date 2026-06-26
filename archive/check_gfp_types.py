"""检查各GFP类型的WT序列和Top 5父本"""
import pandas as pd

df = pd.read_csv('data/processed/full_sequences_full.csv')
print(f'总数据: {len(df)}')
print(f'GFP类型: {df.gfp_type.value_counts().to_dict()}')

for gt in ['cgreGFP', 'ppluGFP', 'amacGFP', 'avGFP']:
    sub = df[df.gfp_type == gt].sort_values('brightness', ascending=False)
    print(f'\n{gt}:')
    print(f'  数量: {len(sub)}')
    print(f'  亮度: [{sub.brightness.min():.4f}, {sub.brightness.max():.4f}]')
    print(f'  Top 5:')
    for i, (_, row) in enumerate(sub.head(5).iterrows()):
        seq = row['sequence']
        b = row['brightness']
        print(f'    #{i+1}: brightness={b:.4f}, len={len(seq)}')
    
    # 检查是否有aaMutations列且为空(WT)
    if 'aaMutations' in sub.columns:
        wt_count = sub[sub['aaMutations'].isna() | (sub['aaMutations'] == 'WT')].shape[0]
        print(f'  WT序列数: {wt_count}')
