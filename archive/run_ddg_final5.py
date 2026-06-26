"""
对最终5条候选序列进行 ESM3-ddG 热稳定性分析
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pandas as pd
import numpy as np
import torch
from transformers import AutoModel
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from src.config import WT_AVGFP, GEN_SEQS_DIR, LOCAL_ESM3_DDG_DIR, AUTHOR, DATE_STR, get_device, seed_everything

# ==================== GFP类型WT序列 ====================
# 各类型的参考WT (用于计算突变位置)
GFP_WT_SEQS = {
    'avGFP': WT_AVGFP,
    'amacGFP': WT_AVGFP,   # amacGFP也是avGFP变体
    'cgreGFP': WT_AVGFP,   # cgreGFP也是avGFP变体
    'ppluGFP': WT_AVGFP,   # ppluGFP也是avGFP变体
}

def load_ddg_model(device):
    model_path = str(LOCAL_ESM3_DDG_DIR) if LOCAL_ESM3_DDG_DIR.exists() else "hazemessam/esm3_ddg_v2"
    print(f"  加载ESM3-ddG: {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()
    tokenizer = EsmSequenceTokenizer()
    return tokenizer, model

def tokenize_seq(tokenizer, seq, device):
    encoded = tokenizer.encode(seq)
    return torch.tensor([encoded], dtype=torch.long, device=device)

def predict_ddg(tokenizer, model, device, wt_seq, mut_seq, positions):
    """预测一组突变的ddG, 逐位点后求和"""
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
        wt_aa = wt_seq[pos]
        mut_aa = mut_seq[pos]
        details.append(f"{wt_aa}{pos+1}{mut_aa}:{val:+.4f}")
    return total_ddg, details

def main():
    seed_everything()
    device = get_device()
    
    print("=" * 60)
    print("  最终5条 ESM3-ddG 热稳定性分析")
    print("=" * 60)
    
    # 1. 加载最终5条
    final_path = GEN_SEQS_DIR / f"final_5_candidates_HCX_{DATE_STR}.csv"
    if not final_path.exists():
        # 尝试搜索
        candidates = sorted(GEN_SEQS_DIR.glob("final_5_candidates_HCX_*.csv"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("未找到最终5条文件!")
            return
        final_path = candidates[0]
    
    df = pd.read_csv(final_path)
    print(f"\n  读取: {final_path.name} ({len(df)}条)")
    
    # 2. 加载模型
    tokenizer, model = load_ddg_model(device)
    
    # 3. 逐条ddG分析
    print(f"\n{'='*60}")
    print("  开始ddG预测 (以avGFP WT为参考)")
    print(f"{'='*60}\n")
    
    results = []
    for i, row in df.iterrows():
        seq = row['sequence']
        gfp_type = row['gfp_type']
        wt_seq = WT_AVGFP
        
        # 找突变位置
        if len(seq) != len(wt_seq):
            print(f"  #{i+1} [{gfp_type}] 序列长度不匹配 ({len(seq)} vs {len(wt_seq)}), 跳过")
            results.append({'total_ddg': float('nan'), 'n_mutations': -1, 'mutation_details': ''})
            continue
        
        mut_positions = [j for j, (a, b) in enumerate(zip(wt_seq, seq)) if a != b]
        n_muts = len(mut_positions)
        
        if n_muts == 0:
            print(f"  #{i+1} [{gfp_type}] 无突变 (与WT相同), ddG=0")
            results.append({'total_ddg': 0.0, 'n_mutations': 0, 'mutation_details': ''})
            continue
        
        # ddG预测
        total_ddg, details = predict_ddg(tokenizer, model, device, wt_seq, seq, mut_positions)
        detail_str = '; '.join(details)
        
        stability = "稳定化" if total_ddg < 0 else "去稳定化"
        print(f"  #{i+1} [{gfp_type}] {n_muts}个突变, total_ddG={total_ddg:+.4f} ({stability})")
        print(f"       加权亮度={row['weighted_avg']:.3f}, 突变: {detail_str}")
        
        results.append({
            'total_ddg': total_ddg,
            'n_mutations': n_muts,
            'mutation_details': detail_str
        })
    
    # 4. 汇总
    res_df = pd.DataFrame(results)
    out_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
    
    out_path = GEN_SEQS_DIR / f"final_5_ddg_analysis_HCX_{DATE_STR}.csv"
    out_df.to_csv(out_path, index=False)
    
    print(f"\n{'='*60}")
    print("  汇总")
    print(f"{'='*60}")
    print(f"{'#':<3} {'类型':<10} {'加权亮度':>8} {'突变数':>6} {'ddG':>10} {'判定':<8}")
    print("-" * 55)
    for i, (r, d) in enumerate(zip(df.itertuples(), results)):
        ddg = d['total_ddg']
        if np.isnan(ddg):
            judgment = "N/A"
            ddg_str = "N/A"
        else:
            judgment = "稳定化" if ddg < 0 else "去稳定化"
            ddg_str = f"{ddg:+.4f}"
        print(f"{i+1:<3} {r.gfp_type:<10} {r.weighted_avg:>8.3f} {d['n_mutations']:>6} {ddg_str:>10} {judgment:<8}")
    
    print(f"\n  保存: {out_path.name}")
    
    # 5. 综合建议
    print(f"\n{'='*60}")
    print("  综合建议 (亮度优先 + ddG辅助)")
    print(f"{'='*60}")
    valid = [(i, r, d) for i, (r, d) in enumerate(zip(df.itertuples(), results)) 
             if not np.isnan(d['total_ddg'])]
    
    # 按亮度排序, 标注ddG
    for i, r, d in sorted(valid, key=lambda x: -x[1].weighted_avg):
        ddg = d['total_ddg']
        flag = ""
        if ddg < -1.0:
            flag = " << 强稳定化"
        elif ddg < 0:
            flag = " < 轻微稳定化"
        elif ddg > 2.0:
            flag = " !! 强去稳定化(需注意)"
        elif ddg > 0:
            flag = " ! 轻微去稳定化"
        print(f"  #{i+1} [{r.gfp_type}] 亮度={r.weighted_avg:.3f}, ddG={ddg:+.4f}{flag}")

if __name__ == "__main__":
    main()
