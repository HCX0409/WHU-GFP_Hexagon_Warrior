"""检查V7通过的序列中是否有非法字符"""
import pandas as pd
import re

df = pd.read_csv('generated_seqs/phase2_v7_filtered_HCX_260623.csv')
print(f"Total: {len(df)}")
print(f"Columns: {list(df.columns)}")

seqs = df['sequence'].tolist()
# Standard 20 amino acids
standard_aa = set("ACDEFGHIKLMNPQRSTVWY")

bad_seqs = []
for i, s in enumerate(seqs):
    chars = set(s)
    non_std = chars - standard_aa
    if non_std:
        bad_seqs.append((i, s[:50], non_std))

print(f"\nSequences with non-standard AA: {len(bad_seqs)}")
for idx, preview, non_std in bad_seqs[:10]:
    print(f"  [{idx}] non_std={non_std}, preview={preview}...")

# Check sequence lengths
lengths = [len(s) for s in seqs]
print(f"\nSequence lengths: min={min(lengths)}, max={max(lengths)}, unique={set(lengths)}")

# Check first few sequences
print(f"\nFirst 3 sequences (first 30 chars):")
for s in seqs[:3]:
    print(f"  {s[:30]}... (len={len(s)})")
