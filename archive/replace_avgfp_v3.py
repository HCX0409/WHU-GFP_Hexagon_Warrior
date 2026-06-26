"""正确替换avGFP: 排除v1和v2的avGFP，取真正的下一个"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import pandas as pd
import numpy as np
import torch
from transformers import AutoModel
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from src.config import WT_AVGFP as WT, GEN_SEQS_DIR, LOCAL_ESM3_DDG_DIR, DATE_STR, get_device, seed_everything

def load_ddg_model(device):
    model = AutoModel.from_pretrained(str(LOCAL_ESM3_DDG_DIR), trust_remote_code=True).to(device)
    model.eval()
    return EsmSequenceTokenizer(), model

def predict_ddg(tokenizer, model, device, wt_seq, mut_seq, positions):
    wt_ids = torch.tensor([tokenizer.encode(wt_seq)], dtype=torch.long, device=device)
    mut_ids = torch.tensor([tokenizer.encode(mut_seq)], dtype=torch.long, device=device)
    total_ddg = 0.0
    details = []
    for pos in positions:
        pos_t = torch.tensor([pos], dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            output = model(wt_ids, mut_ids, positions=pos_t)
            logits = output['logits']
        val = float(logits.sum().cpu())
        total_ddg += val
        details.append(f"{wt_seq[pos]}{pos+1}{mut_seq[pos]}:{val:+.4f}")
    return total_ddg, details

def main():
    seed_everything()
    device = get_device()
    
    all_df = pd.read_csv('generated_seqs/all_scored_merged_HCX_260625.csv')
    excl_seqs = set(pd.read_csv('data/raw/Exclusion_List.csv')['Sequence'].values)
    df = all_df[~all_df['sequence'].isin(excl_seqs)].copy()
    
    # 收集所有已排除的avGFP (v1 + v2)
    v1 = pd.read_csv(GEN_SEQS_DIR / f"final_5_candidates_HCX_{DATE_STR}.csv")
    v2 = pd.read_csv(GEN_SEQS_DIR / f"final_5_candidates_v2_HCX_{DATE_STR}.csv")
    
    exclude_avgfp = set()
    for f in [v1, v2]:
        for _, r in f[f['gfp_type']=='avGFP'].iterrows():
            exclude_avgfp.add(r['sequence'])
    
    print(f"  排除的avGFP序列数: {len(exclude_avgfp)}")
    
    # v2中保留的4条 (cgreGFP, ppluGFP, amacGFP x2)
    kept = v2[v2['gfp_type']!='avGFP'].copy()
    
    # 找avGFP Top候选
    avgfp_cands = df[df['gfp_type']=='avGFP'].copy()
    avgfp_cands = avgfp_cands[~avgfp_cands['sequence'].isin(exclude_avgfp)]
    avgfp_cands = avgfp_cands.sort_values('weighted_avg', ascending=False)
    
    # 显示Top 5供参考
    print(f"\n  avGFP剩余候选Top 5:")
    for rank, (_, row) in enumerate(avgfp_cands.head(5).iterrows()):
        seq = row['sequence']
        muts = [(i+1, WT[i], seq[i]) for i in range(len(WT)) if WT[i] != seq[i]]
        print(f"    #{rank+1}: weighted={row['weighted_avg']:.3f}, muts={', '.join(f'{a}{p}{b}' for p,a,b in muts)}")
    
    # 取Top 1
    new_avgfp = avgfp_cands.iloc[0]
    seq = new_avgfp['sequence']
    muts = [(i+1, WT[i], seq[i]) for i in range(len(WT)) if WT[i] != seq[i]]
    print(f"\n  选用: weighted={new_avgfp['weighted_avg']:.3f}")
    print(f"  突变: {', '.join(f'{a}{p}{b}' for p,a,b in muts)}")
    
    # 组装v3
    new_rows = [r.to_dict() for _, r in kept.iterrows()]
    new_rows.append(new_avgfp.to_dict())
    new_df = pd.DataFrame(new_rows).sort_values('weighted_avg', ascending=False).reset_index(drop=True)
    
    out_path = GEN_SEQS_DIR / f"final_5_candidates_v3_HCX_{DATE_STR}.csv"
    out_cols = [c for c in v2.columns if c in new_df.columns]
    new_df[out_cols].to_csv(out_path, index=False)
    print(f"\n  保存: {out_path.name}")
    
    # ddG
    print(f"\n{'='*70}")
    print("  ddG分析")
    print(f"{'='*70}")
    
    tokenizer, model = load_ddg_model(device)
    mut_positions = [i for i in range(len(WT)) if WT[i] != seq[i]]
    total_ddg, details = predict_ddg(tokenizer, model, device, WT, seq, mut_positions)
    stability = "稳定化" if total_ddg < 0 else "去稳定化"
    print(f"\n  [avGFP] {len(mut_positions)}个突变, ddG={total_ddg:+.4f} ({stability})")
    print(f"  突变细节: {'; '.join(details)}")
    flag = ""
    if total_ddg < -1.0: flag = " << 强稳定化"
    elif total_ddg < 0: flag = " < 轻微稳定化"
    elif total_ddg > 2.0: flag = " !! 强去稳定化"
    elif total_ddg > 0: flag = " ! 轻微去稳定化"
    print(f"  判定: {flag}")
    
    # 汇总
    print(f"\n{'='*70}")
    print("  v3最终5条")
    print(f"{'='*70}")
    print(f"{'#':<3} {'类型':<10} {'加权':>8} {'V7':>8} {'V9':>8} {'V5':>8} {'突变':>6}")
    print("-" * 60)
    for i, r in new_df.iterrows():
        s = r['sequence']
        n = sum(1 for j in range(len(WT)) if j < len(s) and WT[j] != s[j])
        print(f"{i+1:<3} {r['gfp_type']:<10} {r['weighted_avg']:>8.3f} {r['pred_v7']:>8.3f} {r['pred_v9_abs']:>8.3f} {r['pred_v5_abs']:>8.3f} {n:>6}")

if __name__ == "__main__":
    main()
