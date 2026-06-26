"""替换cgreGFP和avGFP候选，生成新的final 5"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pandas as pd
import numpy as np
import torch
from transformers import AutoModel
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from src.config import WT_AVGFP as WT, GEN_SEQS_DIR, LOCAL_ESM3_DDG_DIR, AUTHOR, DATE_STR, get_device, seed_everything

def load_ddg_model(device):
    model_path = str(LOCAL_ESM3_DDG_DIR)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()
    tokenizer = EsmSequenceTokenizer()
    return tokenizer, model

def tokenize_seq(tokenizer, seq, device):
    encoded = tokenizer.encode(seq)
    return torch.tensor([encoded], dtype=torch.long, device=device)

def predict_ddg(tokenizer, model, device, wt_seq, mut_seq, positions):
    wt_ids = tokenize_seq(tokenizer, wt_seq, device)
    mut_ids = tokenize_seq(tokenizer, mut_seq, device)
    total_ddg = 0.0
    details = []
    for pos in positions:
        pos_tensor = torch.tensor([pos], dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            output = model(wt_ids, mut_ids, positions=pos_tensor)
            logits = output['logits']
        val = float(logits.sum().cpu())
        total_ddg += val
        details.append(f"{wt_seq[pos]}{pos+1}{mut_seq[pos]}:{val:+.4f}")
    return total_ddg, details

def main():
    seed_everything()
    device = get_device()
    
    # 加载数据
    all_df = pd.read_csv('generated_seqs/all_scored_merged_HCX_260625.csv')
    final5 = pd.read_csv('generated_seqs/final_5_candidates_HCX_260625.csv')
    
    # 排除列表
    excl_df = pd.read_csv('data/raw/Exclusion_List.csv')
    excl_seqs = set(excl_df['Sequence'].values)
    df = all_df[~all_df['sequence'].isin(excl_seqs)].copy()
    
    exclude_seqs = set(final5['sequence'].values)
    
    print("="*70)
    print("  替换最终5条候选")
    print("="*70)
    
    # 保留3条 (amacGFP x2 + ppluGFP)
    kept = final5[final5['gfp_type'].isin(['amacGFP', 'ppluGFP'])].copy()
    print(f"\n  保留:")
    for _, r in kept.iterrows():
        print(f"    {r['gfp_type']}: weighted={r['weighted_avg']:.3f}")
    
    # 替换cgreGFP: Top 1 = S30F, C48V, E172K (weighted=4.464)
    cgre_cands = df[df['gfp_type']=='cgreGFP']
    cgre_cands = cgre_cands[~cgre_cands['sequence'].isin(exclude_seqs)]
    cgre_cands = cgre_cands.sort_values('weighted_avg', ascending=False)
    new_cgre = cgre_cands.iloc[0]
    print(f"\n  新cgreGFP: weighted={new_cgre['weighted_avg']:.3f}")
    seq = new_cgre['sequence']
    muts = [(i+1, WT[i], seq[i]) for i in range(len(WT)) if WT[i] != seq[i]]
    print(f"    突变: {', '.join(f'{a}{p}{b}' for p,a,b in muts)}")
    
    # 替换avGFP: Top 1 = G104Y, D155E, K166V (weighted=3.710)
    avgfp_cands = df[df['gfp_type']=='avGFP']
    avgfp_cands = avgfp_cands[~avgfp_cands['sequence'].isin(exclude_seqs)]
    avgfp_cands = avgfp_cands.sort_values('weighted_avg', ascending=False)
    new_avgfp = avgfp_cands.iloc[0]
    print(f"\n  新avGFP: weighted={new_avgfp['weighted_avg']:.3f}")
    seq = new_avgfp['sequence']
    muts = [(i+1, WT[i], seq[i]) for i in range(len(WT)) if WT[i] != seq[i]]
    print(f"    突变: {', '.join(f'{a}{p}{b}' for p,a,b in muts)}")
    
    # 组装新final 5
    new_rows = []
    for _, r in kept.iterrows():
        new_rows.append(r.to_dict())
    
    new_rows.append(new_cgre.to_dict())
    new_rows.append(new_avgfp.to_dict())
    
    new_df = pd.DataFrame(new_rows)
    # 按加权分降序
    new_df = new_df.sort_values('weighted_avg', ascending=False).reset_index(drop=True)
    
    # 保存
    out_path = GEN_SEQS_DIR / f"final_5_candidates_v2_HCX_{DATE_STR}.csv"
    out_cols = [c for c in final5.columns if c in new_df.columns]
    new_df[out_cols].to_csv(out_path, index=False)
    print(f"\n  保存: {out_path.name}")
    
    # ddG分析
    print(f"\n{'='*70}")
    print("  对新2条候选跑ddG分析")
    print(f"{'='*70}")
    
    tokenizer, model = load_ddg_model(device)
    
    new_candidates = [
        ("cgreGFP", new_cgre),
        ("avGFP", new_avgfp),
    ]
    
    for label, row in new_candidates:
        seq = row['sequence']
        mut_positions = [i for i in range(len(WT)) if WT[i] != seq[i]]
        
        if len(mut_positions) == 0:
            print(f"\n  [{label}] 无突变, ddG=0")
            continue
        
        total_ddg, details = predict_ddg(tokenizer, model, device, WT, seq, mut_positions)
        stability = "稳定化" if total_ddg < 0 else "去稳定化"
        print(f"\n  [{label}] {len(mut_positions)}个突变, ddG={total_ddg:+.4f} ({stability})")
        print(f"    weighted={row['weighted_avg']:.3f}")
        print(f"    突变细节: {'; '.join(details)}")
        
        flag = ""
        if total_ddg < -1.0: flag = " << 强稳定化"
        elif total_ddg < 0: flag = " < 轻微稳定化"
        elif total_ddg > 2.0: flag = " !! 强去稳定化"
        elif total_ddg > 0: flag = " ! 轻微去稳定化"
        print(f"    判定: {flag}")
    
    # 完整汇总
    print(f"\n{'='*70}")
    print("  新版最终5条汇总")
    print(f"{'='*70}")
    print(f"{'#':<3} {'类型':<10} {'加权亮度':>8} {'V7':>8} {'V9':>8} {'V5':>8} {'突变':>6} {'votes':>6}")
    print("-" * 70)
    for i, r in new_df.iterrows():
        seq = r['sequence']
        n_muts = sum(1 for j in range(len(WT)) if j < len(seq) and WT[j] != seq[j])
        print(f"{i+1:<3} {r['gfp_type']:<10} {r['weighted_avg']:>8.3f} {r['pred_v7']:>8.3f} {r['pred_v9_abs']:>8.3f} {r['pred_v5_abs']:>8.3f} {n_muts:>6} {r.get('votes','?'):>6}")
    
    print(f"\n  文件: {out_path.name}")

if __name__ == "__main__":
    main()
