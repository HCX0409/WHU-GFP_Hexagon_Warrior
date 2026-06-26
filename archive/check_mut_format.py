import pandas as pd

df = pd.read_excel('data/raw/GFP_data.xlsx', sheet_name='brightness')
print('前10条aaMutations:')
for i, mut in enumerate(df['aaMutations'].head(10)):
    print(f"  {i+1}: '{mut}' (type={type(mut).__name__}, len={len(str(mut)) if pd.notna(mut) else 0})")

print(f'\n总行数: {len(df)}')
print(f'非空突变数: {(df["aaMutations"].notna() & (df["aaMutations"] != "")).sum()}')

# 检查是否有逗号分隔
sample = df['aaMutations'].dropna().iloc[0]
print(f'\n第一个非空样本: "{sample}"')
print(f'包含逗号: {"," in str(sample)}')
if ',' in str(sample):
    print(f'分割后: {str(sample).split(",")}')
    print(f'长度: {len(str(sample).split(","))}')
