"""
对20条湿实验验证的高亮度序列用V5/V7/V9分别打分
测试模型在未知GFP类型下的预测能力
"""
import sys, os, gc, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODELS_DIR, GEN_SEQS_DIR, LOCAL_ESM2_DIR

ESM2_MAX_LENGTH = 1024
GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {'avGFP': 0, 'amacGFP': 1, 'cgreGFP': 2, 'ppluGFP': 3}
GFP_WT_BRIGHTNESS = {
    'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258
}

# avGFP WT
WT_AVGFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

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


def extract_embeddings(sequences):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    tok.model_max_length = ESM2_MAX_LENGTH
    esm2 = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR), torch_dtype=torch.float16).to('cuda').eval()
    ds = SeqDataset(sequences)
    dl = DataLoader(ds, batch_size=64, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tok))
    all_embeds = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="ESM-2嵌入"):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = esm2(**batch)
            attn = batch['attention_mask']
            mask = attn.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_embeds.append(pooled.cpu().numpy())
    del esm2
    clear_gpu_cache()
    return np.concatenate(all_embeds, axis=0), tok


def predict_v7(embeddings, gfp_type_idx_list):
    from src.phase1_frozen_esm2_mlp import ResidualBrightnessMLP
    model_dirs = list(MODELS_DIR.glob("ESM2_ResidualMLP_V75_72k_*"))
    if not model_dirs:
        model_dirs = list(MODELS_DIR.glob("ESM2_Frozen_MLP_V7_72k_*"))
    model_dir = sorted(model_dirs)[-1]
    config = joblib.load(model_dir / "model_config.joblib")
    input_dim = config.get('input_dim', 1280)
    model = ResidualBrightnessMLP(input_dim=input_dim, num_gfp_types=4)
    pt_path = model_dir / "regressor.pt"
    if not pt_path.exists():
        pt_path = model_dir / "mlp_model.pt"
    model.load_state_dict(torch.load(pt_path, map_location='cuda', weights_only=False))
    model.to('cuda').eval()

    results = {}
    for gfp_type, type_idx in GFP_TYPE_TO_IDX.items():
        type_arr = np.full(len(embeddings), type_idx, dtype=int)
        emb_t = torch.FloatTensor(embeddings).to('cuda')
        typ_t = torch.LongTensor(type_arr).to('cuda')
        with torch.no_grad():
            _, _, pred_abs = model(emb_t, typ_t)
        results[gfp_type] = pred_abs.cpu().numpy()
    del model
    clear_gpu_cache()
    return results


def predict_v9(sequences, tokenizer):
    from transformers import AutoModel
    from peft import PeftModel
    lora_dir = sorted(MODELS_DIR.glob("ESM2_LoRA_V9_72k_*"))[-1]
    base = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR), torch_dtype=torch.float16).to('cuda')
    lora_model = PeftModel.from_pretrained(base, str(lora_dir))

    class V9Regressor(nn.Module):
        def __init__(self, base_model, num_gfp_types=4):
            super().__init__()
            self.base_model = base_model
            self.pool_norm = nn.LayerNorm(base_model.config.hidden_size)
            input_dim = base_model.config.hidden_size + num_gfp_types
            self.regressor = nn.Sequential(
                nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.25),
                nn.Linear(768, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
    regressor = V9Regressor(lora_model)
    head_state = torch.load(lora_dir / "regressor.pt", map_location='cuda', weights_only=False)
    first_key = list(head_state.keys())[0]
    if first_key.startswith("regressor."):
        rs = {k.replace("regressor.", ""): v for k, v in head_state.items() if k.startswith("regressor.")}
        regressor.regressor.load_state_dict(rs)
    else:
        regressor.regressor.load_state_dict(head_state)
    regressor = regressor.to('cuda').eval()

    results = {}
    for gfp_type, type_idx in GFP_TYPE_TO_IDX.items():
        all_preds = []
        for i in range(0, len(sequences), 64):
            batch_seqs = sequences[i:i+64]
            batch_types = [type_idx] * len(batch_seqs)
            tok_out = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                               truncation=True, max_length=ESM2_MAX_LENGTH)
            tok_out = {k: v.to('cuda') for k, v in tok_out.items()}
            with torch.no_grad():
                outputs = lora_model(**tok_out)
                attn = tok_out['attention_mask']
                mask = attn.unsqueeze(-1).float()
                pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = regressor.pool_norm(pooled)
                onehot = torch.zeros(pooled.size(0), 4, device='cuda')
                onehot.scatter_(1, torch.LongTensor(batch_types).unsqueeze(1).to('cuda'), 1.0)
                combined = torch.cat([pooled.float(), onehot], dim=1)
                preds = regressor.regressor(combined).squeeze(-1)
                # V9 predicts delta, need to add WT brightness
                delta = preds.cpu().numpy()
                abs_val = delta + GFP_WT_BRIGHTNESS[gfp_type]
                all_preds.append(abs_val)
        results[gfp_type] = np.concatenate(all_preds)
    del lora_model, base, tokenizer, regressor
    clear_gpu_cache()
    return results


def predict_v5(sequences, tokenizer):
    from transformers import AutoModel
    from peft import PeftModel
    v5_dir = sorted(MODELS_DIR.glob("ESM2_LoRA_V5_*"))[-1]
    base = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR), torch_dtype=torch.float16).to('cuda')
    v5_model = PeftModel.from_pretrained(base, str(v5_dir))

    class V5Regressor(nn.Module):
        def __init__(self, base_model, num_gfp_types=4):
            super().__init__()
            self.base_model = base_model
            input_dim = base_model.config.hidden_size + num_gfp_types
            self.regressor = nn.Sequential(
                nn.Linear(input_dim, 768), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
    regressor = V5Regressor(v5_model)
    head_state = torch.load(v5_dir / "regressor.pt", map_location='cuda', weights_only=False)
    first_key = list(head_state.keys())[0]
    if first_key.startswith("regressor."):
        rs = {k.replace("regressor.", ""): v for k, v in head_state.items() if k.startswith("regressor.")}
        regressor.regressor.load_state_dict(rs)
    else:
        regressor.regressor.load_state_dict(head_state)
    regressor = regressor.to('cuda').eval()

    results = {}
    for gfp_type, type_idx in GFP_TYPE_TO_IDX.items():
        all_preds = []
        for i in range(0, len(sequences), 64):
            batch_seqs = sequences[i:i+64]
            batch_types = [type_idx] * len(batch_seqs)
            tok_out = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                               truncation=True, max_length=ESM2_MAX_LENGTH)
            tok_out = {k: v.to('cuda') for k, v in tok_out.items()}
            with torch.no_grad():
                outputs = v5_model(**tok_out)
                attn = tok_out['attention_mask']
                mask = attn.unsqueeze(-1).float()
                pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                onehot = torch.zeros(pooled.size(0), 4, device='cuda')
                onehot.scatter_(1, torch.LongTensor(batch_types).unsqueeze(1).to('cuda'), 1.0)
                combined = torch.cat([pooled.float(), onehot], dim=1)
                preds = regressor.regressor(combined).squeeze(-1)
                delta = preds.cpu().numpy()
                abs_val = delta + GFP_WT_BRIGHTNESS[gfp_type]
                all_preds.append(abs_val)
        results[gfp_type] = np.concatenate(all_preds)
    del v5_model, base, tokenizer, regressor
    clear_gpu_cache()
    return results


def hamming_dist(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))


def main():
    # 加载数据
    xls = pd.ExcelFile(PROJECT_ROOT / 'data' / 'raw' / 'GFP_data.xlsx')
    df = pd.read_excel(xls, 'beforetopseqs')
    sequences = df['sequence'].tolist()
    print(f"湿实验序列: {len(sequences)} 条, 长度={len(sequences[0])}")

    # 与WT的汉明距离
    print(f"\n与avGFP WT的汉明距离:")
    for i, seq in enumerate(sequences):
        dist = hamming_dist(seq, WT_AVGFP)
        print(f"  #{i}: {dist} mutations")

    # 1. 提取嵌入
    print(f"\n{'='*60}")
    print(f" 提取ESM-2嵌入")
    print(f"{'='*60}")
    from transformers import AutoTokenizer
    embeddings, tokenizer = extract_embeddings(sequences)

    # 2. V7预测 (4种类型)
    print(f"\n{'='*60}")
    print(f" V7预测 (各类型)")
    print(f"{'='*60}")
    v7_results = predict_v7(embeddings, None)

    # 3. V9预测
    print(f"\n{'='*60}")
    print(f" V9预测 (各类型)")
    print(f"{'='*60}")
    from transformers import AutoTokenizer as AT2
    tok2 = AT2.from_pretrained(str(LOCAL_ESM2_DIR))
    tok2.model_max_length = ESM2_MAX_LENGTH
    v9_results = predict_v9(sequences, tok2)

    # 4. V5预测
    print(f"\n{'='*60}")
    print(f" V5预测 (各类型)")
    print(f"{'='*60}")
    tok3 = AT2.from_pretrained(str(LOCAL_ESM2_DIR))
    tok3.model_max_length = ESM2_MAX_LENGTH
    v5_results = predict_v5(sequences, tok3)

    # 5. 汇总输出
    print(f"\n{'='*60}")
    print(f" 汇总: 各模型 × 各GFP类型 预测亮度")
    print(f"{'='*60}")

    # 逐序列输出
    for i in range(len(sequences)):
        print(f"\n  序列 #{i} (year={df.iloc[i]['year']}):")
        print(f"    {'类型':>8}  {'V5':>6}  {'V7':>6}  {'V9':>6}  {'加权平均':>8}")
        best_avg = 0
        best_type = ''
        for gt in GFP_TYPE_LIST:
            v5 = v5_results[gt][i]
            v7 = v7_results[gt][i]
            v9 = v9_results[gt][i]
            avg = 0.15*v5 + 0.35*v7 + 0.50*v9
            if avg > best_avg:
                best_avg = avg
                best_type = gt
            print(f"    {gt:>8}  {v5:6.3f}  {v7:6.3f}  {v9:6.3f}  {avg:8.3f}")
        print(f"    >>> 最佳类型: {best_type}, 加权平均: {best_avg:.3f}")

    # 统计摘要
    print(f"\n{'='*60}")
    print(f" 统计摘要 (最佳类型)")
    print(f"{'='*60}")
    best_types = []
    best_avgs = []
    for i in range(len(sequences)):
        avgs = {}
        for gt in GFP_TYPE_LIST:
            avgs[gt] = 0.15*v5_results[gt][i] + 0.35*v7_results[gt][i] + 0.50*v9_results[gt][i]
        bt = max(avgs, key=avgs.get)
        best_types.append(bt)
        best_avgs.append(avgs[bt])

    print(f"  最佳类型分布: {pd.Series(best_types).value_counts().to_dict()}")
    print(f"  加权平均范围: [{min(best_avgs):.3f}, {max(best_avgs):.3f}]")
    print(f"  加权平均均值: {np.mean(best_avgs):.3f}")
    print(f"  加权平均>4.0: {sum(1 for x in best_avgs if x>=4.0)}/{len(best_avgs)}")
    print(f"  加权平均>4.2: {sum(1 for x in best_avgs if x>=4.2)}/{len(best_avgs)}")


if __name__ == "__main__":
    main()
