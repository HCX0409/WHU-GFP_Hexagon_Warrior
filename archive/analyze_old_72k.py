"""
分析旧72k数据中多突变样本的实际分布
（虽然n_mutations列错误，但可以从序列推导真实突变数）
"""
import pandas as pd
from src.phase1_data import parse_mutations

WT_AVGFP = ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"
             "LSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGI"
             "DFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGP"
             "VLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK")

# 读取原始141k数据重新计算真实突变数
raw_df = pd.read_excel('data/raw/GFP_data.xlsx', sheet_name='brightness')
print(f"原始数据: {len(raw_df)} 条\n")

# 解析并计算真实突变数
records = []
for idx, row in raw_df.iterrows():
    mut_str = row['aaMutations']
    if pd.notna(mut_str) and str(mut_str).upper() not in ('WT', ''):
        # 真实突变数（用冒号分隔）
        true_n_mut = len(str(mut_str).split(':'))
        records.append({
            'gfp_type': row['GFP type'],
            'brightness': row['Brightness'],
            'true_n_mutations': true_n_mut
        })

true_df = pd.DataFrame(records)
print(f"成功解析: {len(true_df)} 条\n")

# 统计真实突变数分布
print("真实突变数分布:")
print(true_df['true_n_mutations'].value_counts().sort_index().head(15))
print(f"\n统计:")
print(f"  均值: {true_df['true_n_mutations'].mean():.1f}")
print(f"  >=3突变: {(true_df['true_n_mutations'] >= 3).sum()} ({(true_df['true_n_mutations'] >= 3).sum()/len(true_df)*100:.1f}%)")

# 模拟旧72k的筛选结果（假设n_mut全是1）
print("\n" + "="*60)
print("模拟旧72k筛选（n_mut全是1时的情况）:")
print("="*60)

# 计算delta
WT_BRIGHTNESS = {'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258}
true_df['delta'] = true_df.apply(lambda r: r['brightness'] - WT_BRIGHTNESS.get(r['gfp_type'], 3.7192), axis=1)

# Tier定义（假设n_mut全是1）
true_df['simulated_tier'] = 4
true_df.loc[abs(true_df['delta']) > 0.5, 'simulated_tier'] = 1
true_df.loc[(true_df['delta'] > 0.3) & (true_df['simulated_tier'] > 1), 'simulated_tier'] = 2
# Tier 3失效（因为假设n_mut全是1）

print(f"Tier 1: {(true_df['simulated_tier'] == 1).sum()} 条")
print(f"Tier 2: {(true_df['simulated_tier'] == 2).sum()} 条")
print(f"Tier 3: {(true_df['simulated_tier'] == 3).sum()} 条 ← 应该是0")
print(f"Tier 4: {(true_df['simulated_tier'] == 4).sum()} 条")

# 检查Tier 1和4中实际包含多少多突变样本
tier1_multi = true_df[(true_df['simulated_tier'] == 1) & (true_df['true_n_mutations'] >= 3)]
tier4_multi = true_df[(true_df['simulated_tier'] == 4) & (true_df['true_n_mutations'] >= 3)]
print(f"\nTier 1中>=3突变: {len(tier1_multi)} 条 ({len(tier1_multi)/(true_df['simulated_tier']==1).sum()*100:.1f}%)")
print(f"Tier 4中>=3突变: {len(tier4_multi)} 条 ({len(tier4_multi)/(true_df['simulated_tier']==4).sum()*100:.1f}%)")
print(f"总计>=3突变被保留: {len(tier1_multi) + len(tier4_multi)} 条")
