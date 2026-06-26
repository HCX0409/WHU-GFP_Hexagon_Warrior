"""快速诊断: amacGFP和avGFP的V7预测分布"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import LOCAL_ESM2_DIR, RAW_DIR, WT_AVGFP, RANDOM_SEED, get_device
from src.phase2_generate_hierarchical import (
    ResidualBrightnessMLP, load_v7_model, extract_esm2_embeddings,
    predict_v7, parse_mutations, generate_variants,
    ALL_GFP_TYPES, GFP_TYPE_TO_IDX, N_VARIANTS_PER_PARENT
)

# 加载Top 50父本
raw_df = pd.read_excel(RAW_DIR / "GFP_data.xlsx", sheet_name="brightness")
parent_records = []
for gfp_type in ['amacGFP', 'avGFP']:
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
print(f"父本: {len(parent_df)} 条")

# 生成变体(只取一小批做诊断, 2000条/类型足够看分布)
import numpy as np
rng = np.random.default_rng(RANDOM_SEED + 1000)
variant_df = generate_variants(parent_df, seed=RANDOM_SEED + 1000)
# 只取前4000条做诊断
variant_df = variant_df.head(4000)
variant_df = variant_df.drop_duplicates(subset=['sequence']).reset_index(drop=True)
print(f"变体: {len(variant_df)} 条")

# 加载模型做V7预测
v7_model, v7_device = load_v7_model()
esm2_base, esm2_tok, esm2_dev = torch.nn.Module(), None, None  # placeholder

from src.phase2_generate_hierarchical import load_esm2_base_for_embeddings
esm2_base, esm2_tok, esm2_dev = load_esm2_base_for_embeddings()
embeddings = extract_esm2_embeddings(esm2_base, esm2_tok, esm2_dev, variant_df['sequence'].tolist())
variant_df['pred_v7'] = predict_v7(v7_model, v7_device, embeddings, variant_df['type_idx'].values)

# 分析分布
for gt in ['amacGFP', 'avGFP']:
    sub = variant_df[variant_df['gfp_type'] == gt]
    print(f"\n{'='*50}")
    print(f"{gt}: {len(sub)} 条变体")
    print(f"  V7统计: mean={sub.pred_v7.mean():.3f}, std={sub.pred_v7.std():.3f}")
    print(f"  V7范围: [{sub.pred_v7.min():.3f}, {sub.pred_v7.max():.3f}]")
    print(f"  分位数:")
    for p in [99, 95, 90, 80, 70, 50, 25, 10]:
        val = np.percentile(sub.pred_v7.values, p)
        print(f"    P{p}: {val:.3f}")
    # 不同阈值的通过数
    print(f"  不同阈值通过数:")
    for thr in [3.87, 3.80, 3.70, 3.60, 3.50, 3.40, 3.30, 3.20, 3.00]:
        n = (sub.pred_v7 >= thr).sum()
        print(f"    >={thr}: {n} ({n/len(sub)*100:.1f}%)")
