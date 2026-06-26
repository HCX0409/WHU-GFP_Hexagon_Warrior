"""
对旧Phase 2高亮度候选序列用V9+V7重新交叉验证打分
输入: phase2_bright_mut_5000_HCX_260611.csv (V5已打分)
输出: V5+V7+V9三模型加权评分结果
"""
import sys, os, gc, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    MODELS_DIR, GEN_SEQS_DIR, LOCAL_ESM2_DIR, AUTHOR, DATE_STR
)

ESM2_MAX_LENGTH = 1024
BATCH_SIZE = 64
GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {'avGFP': 0, 'amacGFP': 1, 'cgreGFP': 2, 'ppluGFP': 3}
GFP_WT_BRIGHTNESS = {
    'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258
}

V5_THRESHOLD = 4.3
MODEL_WEIGHTS = {'v5': 0.15, 'v7': 0.35, 'v9': 0.50}


def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class SeqDataset(Dataset):
    def __init__(self, seqs): self.seqs = list(seqs)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx]

def collate_fn(batch, tokenizer):
    return {k: v for k, v in tokenizer(batch, return_tensors="pt", padding=True,
            truncation=True, max_length=ESM2_MAX_LENGTH).items()}

def extract_embeddings(esm2, tok, device, sequences):
    ds = SeqDataset(sequences)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tok))
    all_embeds = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="ESM-2嵌入"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = esm2(**batch)
            attn = batch['attention_mask']
            mask = attn.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_embeds.append(pooled.cpu().numpy())
    return np.concatenate(all_embeds, axis=0)


def predict_v7(embeddings, gfp_type_indices):
    from src.phase1_frozen_esm2_mlp import ResidualBrightnessMLP
    model_dirs = list(MODELS_DIR.glob("ESM2_ResidualMLP_V75_72k_*"))
    if not model_dirs:
        model_dirs = list(MODELS_DIR.glob("ESM2_Frozen_MLP_V7_72k_*"))
    model_dir = sorted(model_dirs)[-1]
    print(f"  V7模型: {model_dir.name}")
    config = joblib.load(model_dir / "model_config.joblib")
    input_dim = config.get('input_dim', 1280)
    model = ResidualBrightnessMLP(input_dim=input_dim, num_gfp_types=4)
    pt_path = model_dir / "regressor.pt"
    if not pt_path.exists():
        pt_path = model_dir / "mlp_model.pt"
    model.load_state_dict(torch.load(pt_path, map_location='cuda', weights_only=False))
    model.to('cuda').eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(embeddings), 256):
            emb = torch.FloatTensor(embeddings[i:i+256]).to('cuda')
            typ = torch.LongTensor(gfp_type_indices[i:i+256]).to('cuda')
            _, _, pred_abs = model(emb, typ)
            all_preds.append(pred_abs.cpu().numpy())
    del model
    clear_gpu_cache()
    return np.concatenate(all_preds)


class ESM2LoRARegressorV9(nn.Module):
    def __init__(self, base_model, num_gfp_types=4):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.pool_norm = nn.LayerNorm(hidden_size)
        input_dim = hidden_size + num_gfp_types
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.num_gfp_types = num_gfp_types

def predict_v9(sequences, gfp_type_indices):
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    lora_dir = sorted(MODELS_DIR.glob("ESM2_LoRA_V9_72k_*"))[-1]
    print(f"  V9模型: {lora_dir.name}")
    base = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR), torch_dtype=torch.float16).to('cuda')
    tok = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    tok.model_max_length = ESM2_MAX_LENGTH
    lora_model = PeftModel.from_pretrained(base, str(lora_dir))
    metrics = joblib.load(lora_dir / "metrics.joblib")
    output_dim = metrics.get('output_dim', 1)
    regressor = ESM2LoRARegressorV9(lora_model)
    head_state = torch.load(lora_dir / "regressor.pt", map_location='cuda', weights_only=False)
    first_key = list(head_state.keys())[0]
    if first_key.startswith("regressor."):
        rs = {k.replace("regressor.", ""): v for k, v in head_state.items() if k.startswith("regressor.")}
        regressor.regressor.load_state_dict(rs)
    else:
        regressor.regressor.load_state_dict(head_state)
    regressor = regressor.to('cuda').eval()

    ds = SeqDataset(sequences)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tok))
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="V9预测"):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = lora_model(**batch)
            attn = batch['attention_mask']
            mask = attn.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = regressor.pool_norm(pooled)
            type_onehot = torch.zeros(pooled.size(0), 4, device='cuda')
            # 需要当前batch的gfp_type_idx
            all_preds.append(regressor.regressor(
                torch.cat([pooled.float(), type_onehot], dim=1)
            ).cpu().numpy())
    del lora_model, base, tok, regressor
    clear_gpu_cache()
    return np.concatenate(all_preds).flatten()


def predict_v9_v2(sequences, gfp_type_indices):
    """正确的V9预测：逐batch传入gfp_type"""
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    lora_dir = sorted(MODELS_DIR.glob("ESM2_LoRA_V9_72k_*"))[-1]
    print(f"  V9模型: {lora_dir.name}")
    base = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR), torch_dtype=torch.float16).to('cuda')
    tok = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    tok.model_max_length = ESM2_MAX_LENGTH
    lora_model = PeftModel.from_pretrained(base, str(lora_dir))
    regressor = ESM2LoRARegressorV9(lora_model)
    head_state = torch.load(lora_dir / "regressor.pt", map_location='cuda', weights_only=False)
    first_key = list(head_state.keys())[0]
    if first_key.startswith("regressor."):
        rs = {k.replace("regressor.", ""): v for k, v in head_state.items() if k.startswith("regressor.")}
        regressor.regressor.load_state_dict(rs)
    else:
        regressor.regressor.load_state_dict(head_state)
    regressor = regressor.to('cuda').eval()

    all_preds = []
    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="V9预测"):
        batch_seqs = sequences[i:i+BATCH_SIZE]
        batch_types = gfp_type_indices[i:i+BATCH_SIZE]
        tok_out = tok(batch_seqs, return_tensors="pt", padding=True,
                      truncation=True, max_length=ESM2_MAX_LENGTH)
        tok_out = {k: v.to('cuda') for k, v in tok_out.items()}
        with torch.no_grad():
            outputs = lora_model(**tok_out)
            attn = tok_out['attention_mask']
            mask = attn.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = regressor.pool_norm(pooled)
            type_onehot = torch.zeros(pooled.size(0), 4, device='cuda')
            type_onehot.scatter_(1, torch.LongTensor(batch_types).unsqueeze(1).to('cuda'), 1.0)
            combined = torch.cat([pooled.float(), type_onehot], dim=1)
            preds = regressor.regressor(combined).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
    del lora_model, base, tok, regressor
    clear_gpu_cache()
    return np.concatenate(all_preds)


def main():
    print("=" * 60)
    print(" Phase 2旧数据 V9+V7交叉验证")
    print("=" * 60)

    in_path = GEN_SEQS_DIR / "phase2_bright_mut_5000_HCX_260611.csv"
    df = pd.read_csv(in_path)
    print(f"  原始数据: {len(df)} 条")
    print(f"  V5预测范围: [{df.pred_brightness.min():.3f}, {df.pred_brightness.max():.3f}]")

    high_df = df[df.pred_brightness >= V5_THRESHOLD].copy()
    high_df = high_df.sort_values('pred_brightness', ascending=False).reset_index(drop=True)
    print(f"  V5 >= {V5_THRESHOLD}: {len(high_df)} 条")
    print(f"  GFP类型: {high_df.gfp_type.value_counts().to_dict()}")

    sequences = high_df['sequence'].tolist()
    gfp_types = high_df['gfp_type'].tolist()
    gfp_type_indices = [GFP_TYPE_TO_IDX[t] for t in gfp_types]

    # === ESM-2嵌入 ===
    from transformers import AutoTokenizer, AutoModel
    print(f"\n{'='*60}")
    print(f" 提取ESM-2嵌入 ({len(sequences)} 条)")
    print(f"{'='*60}")
    tok = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    tok.model_max_length = ESM2_MAX_LENGTH
    esm2 = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR), torch_dtype=torch.float16).to('cuda').eval()
    embeddings = extract_embeddings(esm2, tok, 'cuda', sequences)
    print(f"  嵌入shape: {embeddings.shape}")
    del esm2
    clear_gpu_cache()

    # === V7 ===
    print(f"\n{'='*60}")
    print(f" V7预测")
    print(f"{'='*60}")
    pred_v7 = predict_v7(embeddings, gfp_type_indices)
    high_df['pred_v7'] = pred_v7
    print(f"  V7范围: [{pred_v7.min():.3f}, {pred_v7.max():.3f}]")
    print(f"  V7>=4.2: {(pred_v7>=4.2).sum()}, V7>=4.3: {(pred_v7>=4.3).sum()}")

    # === V9 ===
    print(f"\n{'='*60}")
    print(f" V9预测")
    print(f"{'='*60}")
    pred_v9_delta = predict_v9_v2(sequences, gfp_type_indices)
    pred_v9_abs = np.array([d + GFP_WT_BRIGHTNESS[t] for d, t in zip(pred_v9_delta, gfp_types)])
    high_df['pred_v9_delta'] = pred_v9_delta
    high_df['pred_v9_abs'] = pred_v9_abs
    print(f"  V9 abs范围: [{pred_v9_abs.min():.3f}, {pred_v9_abs.max():.3f}]")
    print(f"  V9>=4.3: {(pred_v9_abs>=4.3).sum()}, V9>=4.4: {(pred_v9_abs>=4.4).sum()}")

    # === 三模型加权 ===
    print(f"\n{'='*60}")
    print(f" 三模型加权评分")
    print(f"{'='*60}")
    w = MODEL_WEIGHTS
    high_df['pred_v5'] = high_df['pred_brightness'].values
    high_df['weighted_avg'] = (
        w['v5'] * high_df['pred_v5'] +
        w['v7'] * high_df['pred_v7'] +
        w['v9'] * high_df['pred_v9_abs']
    )
    for thr_name, thr in [('4.3', 4.3), ('4.2', 4.2)]:
        high_df[f'votes_{thr_name}'] = (
            (high_df['pred_v5'] >= thr).astype(int) +
            (high_df['pred_v7'] >= thr).astype(int) +
            (high_df['pred_v9_abs'] >= thr).astype(int)
        )

    high_df = high_df.sort_values('weighted_avg', ascending=False)

    print(f"\n  加权平均: [{high_df['weighted_avg'].min():.3f}, {high_df['weighted_avg'].max():.3f}]")
    for t in [4.3, 4.2, 4.1, 4.0]:
        print(f"  加权>={t}: {(high_df['weighted_avg']>=t).sum()} 条")

    print(f"\n  共识票数分布 (阈值4.3):")
    for v in range(4):
        print(f"    {v}票: {(high_df['votes_4.3']==v).sum()} 条")

    print(f"\n  共识票数分布 (阈值4.2):")
    for v in range(4):
        print(f"    {v}票: {(high_df['votes_4.2']==v).sum()} 条")

    # Top 30
    print(f"\n{'='*60}")
    print(f" Top 30")
    print(f"{'='*60}")
    print(f"  {'#':>3} {'类型':>8} {'加权':>6} {'V5':>6} {'V7':>6} {'V9':>6} {'票4.3':>5} {'票4.2':>5} {'父本':>6}")
    for i, (_, row) in enumerate(high_df.head(30).iterrows(), 1):
        print(f"  {i:3d} {row['gfp_type']:>8} {row['weighted_avg']:6.3f} "
              f"{row['pred_v5']:6.3f} {row['pred_v7']:6.3f} {row['pred_v9_abs']:6.3f} "
              f"{int(row['votes_4.3']):5d} {int(row['votes_4.2']):5d} {row['parent_brightness']:6.3f}")

    # 保存
    out_cols = ['sequence', 'gfp_type', 'num_mutations_from_parent', 'parent_brightness',
                'pred_v5', 'pred_v7', 'pred_v9_abs', 'pred_v9_delta',
                'weighted_avg', 'votes_4.3', 'votes_4.2',
                'parent_mutations', 'parent_sequence']
    out_path = GEN_SEQS_DIR / f"phase2_crossval_{AUTHOR}_{DATE_STR}.csv"
    save_cols = [c for c in out_cols if c in high_df.columns]
    high_df[save_cols].to_csv(out_path, index=False)
    print(f"\n  已保存: {out_path} ({len(high_df)} 条)")


if __name__ == "__main__":
    main()
