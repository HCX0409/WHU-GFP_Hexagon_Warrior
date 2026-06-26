"""直接分析 beforetopseqs 20条序列的突变特征和类型"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pandas as pd
from src.config import WT_AVGFP as WT

df = pd.read_excel('data/raw/GFP_data.xlsx', sheet_name='beforetopseqs')

# avGFP WT chromophore: S65-Y66-G67 (0-based: 64-65-66)
# Key residue positions (1-based): 65, 66, 67, 145, 146, 147, 148, 203, 206
# 0-based: 64, 65, 66, 144, 145, 146, 147, 202, 205

KEY_POS = {
    64: 'Chr1(65)',    # Chromophore residue 1 (Ser in WT)
    65: 'Chr2(66)',    # Chromophore residue 2 (Tyr in WT) 
    66: 'Chr3(67)',    # Chromophore residue 3 (Gly in WT)
    144: 'b-strand7(145)',
    145: 'b-strand7(146)',
    146: 'b-strand7(147)',
    147: 'b-strand7(148)',
    148: 'b-strand7(149)',
    202: 'Helix(203)',  # T203
    205: 'Dimer(206)',  # A206 = monomeric marker
    162: 'K162',
    166: 'K166',
}

print("="*80)
print("  beforetopseqs 20条序列分析")
print("="*80)

# 先检查排除列表
excl_df = pd.read_csv('data/raw/Exclusion_List.csv')
excl_seqs = set(excl_df['Sequence'].values)
print(f"\n  Exclusion List: {len(excl_seqs)}条序列")

# 检查哪些在排除列表中
in_excl = []
for idx, row in df.iterrows():
    seq = row['sequence']
    if seq in excl_seqs:
        in_excl.append(idx+1)

if in_excl:
    print(f"  在排除列表中的: #{in_excl}")
else:
    print(f"  无序列出现在排除列表中")

# 也检查最终5条
final5 = pd.read_csv('generated_seqs/final_5_candidates_HCX_260625.csv')
print(f"\n  检查最终5条是否在排除列表中:")
for i, row in final5.iterrows():
    in_list = row['sequence'] in excl_seqs
    print(f"    #{i+1} [{row['gfp_type']}]: {'!! 在排除列表中 !!' if in_list else 'OK'}")

print(f"\n{'='*80}")
print("  详细突变分析")
print("="*80)

type_counts = {}
for idx, row in df.iterrows():
    seq = row['sequence']
    year = row['year']
    
    # 找突变
    muts = []
    for i in range(min(len(WT), len(seq))):
        if WT[i] != seq[i]:
            muts.append((i+1, WT[i], seq[i]))  # 1-based position
    
    # 关键位点分析
    chromophore = f"{seq[64]}{seq[65]}{seq[66]}" if len(seq) > 66 else "?"
    key_info = []
    for pos, label in sorted(KEY_POS.items()):
        if pos < len(seq) and pos < len(WT) and seq[pos] != WT[pos]:
            key_info.append(f"{WT[pos]}{pos+1}{seq[pos]}")
    
    # 判断类型 (基于已知GFP变体特征)
    s65 = seq[64] if len(seq) > 64 else '?'
    y66 = seq[65] if len(seq) > 65 else '?'
    g67 = seq[66] if len(seq) > 66 else '?'
    t203 = seq[202] if len(seq) > 202 else '?'
    a206 = seq[205] if len(seq) > 205 else '?'
    
    # 分类逻辑
    seq_type = "avGFP-like"
    if s65 == 'T':
        seq_type = "EGFP-like (S65T)"
    elif s65 == 'C':
        seq_type = "CFP-like (S65C)"
    elif s65 == 'G' or y66 == 'F':
        seq_type = "YFP-like"
    elif s65 == 'L':
        seq_type = "BFP-like (S65L)"
    
    # 单体标记
    if a206 == 'K':
        seq_type += " + monomeric(A206K)"
    elif a206 == 'V':
        seq_type += " + monomeric(A206V)"
    
    type_counts[seq_type] = type_counts.get(seq_type, 0) + 1
    
    in_excl_mark = " [EXCLUDED]" if seq in excl_seqs else ""
    
    print(f"\n  #{idx+1} [{year}] {len(muts)}个突变 {in_excl_mark}")
    print(f"    类型判定: {seq_type}")
    print(f"    发色团(65-67): {chromophore} (WT: SYG)")
    
    mut_str = ', '.join(f'{a}{p}{b}' for p,a,b in muts)
    print(f"    全部突变: {mut_str}")
    if key_info:
        print(f"    关键位点: {', '.join(key_info)}")

# 汇总
print(f"\n{'='*80}")
print("  类型分布汇总")
print("="*80)
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}条")

# 与最终5条对比
print(f"\n{'='*80}")
print("  与最终5条候选对比")
print("="*80)

for i, row in final5.iterrows():
    seq = row['sequence']
    muts = [(j+1, WT[j], seq[j]) for j in range(min(len(WT), len(seq))) if WT[j] != seq[j]]
    s65 = seq[64] if len(seq) > 64 else '?'
    
    print(f"\n  最终#{i+1} [{row['gfp_type']}] weighted={row['weighted_avg']:.3f}")
    print(f"    发色团: {seq[64]}{seq[65]}{seq[66]}")
    print(f"    {len(muts)}个突变: {', '.join(f'{a}{p}{b}' for p,a,b in muts[:10])}")
    
    # 检查是否与beforetopseqs中任何一条相同
    matches = []
    for j, brow in df.iterrows():
        if brow['sequence'] == seq:
            matches.append(j+1)
    if matches:
        print(f"    ** 与beforetopseqs #{matches} 相同! **")
    
    # 检查相似度
    best_sim = 0
    best_match = -1
    for j, brow in df.iterrows():
        bseq = brow['sequence']
        if len(bseq) != len(seq):
            continue
        sim = sum(1 for a,b in zip(seq, bseq) if a == b) / len(seq)
        if sim > best_sim:
            best_sim = sim
            best_match = j + 1
    print(f"    与beforetopseqs最高相似度: #{best_match} ({best_sim:.3%})")
