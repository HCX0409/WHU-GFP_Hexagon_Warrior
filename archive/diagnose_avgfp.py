"""快速诊断avGFP的V7分布"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd, numpy as np, torch
from src.config import RAW_DIR, WT_AVGFP, RANDOM_SEED
from src.phase2_generate_hierarchical import (
    load_v7_model, load_esm2_base_for_embeddings, extract_esm2_embeddings,
    predict_v7, parse_mutations, generate_variants, GFP_TYPE_TO_IDX
)

raw_df = pd.read_excel(RAW_DIR / "GFP_data.xlsx", sheet_name="brightness")
parent_records = []
for gfp_type in ['avGFP']:
    type_idx = GFP_TYPE_TO_IDX[gfp_type]
    type_df = raw_df[raw_df['GFP type'].str.contains(gfp_type, case=False, na=False)]
    type_df = type_df.sort_values('Brightness', ascending=False).head(50)
    for rank, (_, row) in enumerate(type_df.iterrows(), start=1):
        full_seq = parse_mutations(row['aaMutations'], WT_AVGFP)
        if full_seq and len(full_seq) == len(WT_AVGFP):
            parent_records.append({
                'sequence': full_seq, 'brightness': row['Brightness'],
                'gfp_type': gfp_type, 'type_idx': type_idx,
                'rank': rank, 'mutations': str(row['aaMutations'])
            })

parent_df = pd.DataFrame(parent_records)
print(f"avGFP父本: {len(parent_df)} 条")

variant_df = generate_variants(parent_df, seed=RANDOM_SEED + 1000)
variant_df = variant_df.head(4000).drop_duplicates(subset=['sequence']).reset_index(drop=True)
print(f"avGFP变体: {len(variant_df)} 条")

v7_model, v7_device = load_v7_model()
esm2_base, esm2_tok, esm2_dev = load_esm2_base_for_embeddings()
embeddings = extract_esm2_embeddings(esm2_base, esm2_tok, esm2_dev, variant_df['sequence'].tolist())
variant_df['pred_v7'] = predict_v7(v7_model, v7_device, embeddings, variant_df['type_idx'].values)

sub = variant_df
print(f"\navGFP: {len(sub)} 条变体")
print(f"  V7统计: mean={sub.pred_v7.mean():.3f}, std={sub.pred_v7.std():.3f}")
print(f"  V7范围: [{sub.pred_v7.min():.3f}, {sub.pred_v7.max():.3f}]")
print(f"  分位数:")
for p in [99, 95, 90, 80, 70, 50, 25, 10]:
    print(f"    P{p}: {np.percentile(sub.pred_v7.values, p):.3f}")
print(f"  不同阈值通过数:")
for thr in [3.62, 3.50, 3.40, 3.30, 3.20, 3.10, 3.00, 2.90, 2.80]:
    n = (sub.pred_v7 >= thr).sum()
    print(f"    >={thr}: {n} ({n/len(sub)*100:.1f}%)")
