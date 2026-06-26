import pandas as pd

df = pd.read_csv('data/processed/v7_72k_quality_data.csv')
print(f'总样本: {len(df)}')
print(f'\nn_mutations分布 (前15):')
print(df['n_mutations'].value_counts().sort_index().head(15))
print(f'\n突变数统计:')
print(f'  均值: {df["n_mutations"].mean():.1f}')
print(f'  中位数: {df["n_mutations"].median():.0f}')
print(f'  最大值: {df["n_mutations"].max()}')
print(f'  最小值: {df["n_mutations"].min()}')
print(f'  >=3突变的样本: {(df["n_mutations"] >= 3).sum()} ({(df["n_mutations"] >= 3).sum()/len(df)*100:.1f}%)')

# 检查几行示例
print(f'\n示例数据 (前5条):')
for i in range(5):
    print(f"  {i+1}: n_mut={df.iloc[i]['n_mutations']}, delta={df.iloc[i]['delta']:.3f}, brightness={df.iloc[i]['brightness']:.3f}")
