"""
Phase 2: 分层突变 + V7初筛 + V5/V7/V9共识筛选

核心策略:
1. 4种GFP类型全做 (cgreGFP/ppluGFP/amacGFP/avGFP)
2. 各类型独立阈值（接近训练数据最高亮度，不追求超越）
3. 父本: 各类型Top 50最亮序列(4×50=200条父本), 每条500个变体
4. V7初筛: 各类型WT亮度-0.1（筛掉明显差的）
5. V5+V9精筛, 三模型共识投票(>=2/3达标)
6. 5轮迭代(每轮不同随机种子), 累计合并去重

产出:
  generated_seqs/phase2_hierarchical_{AUTHOR}_{DATE_STR}.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
import gc
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from src.config import (
    GEN_SEQS_DIR, MODELS_DIR, LOCAL_ESM2_DIR, RAW_DIR, PROCESSED_DIR,
    WT_AVGFP, AUTHOR, DATE_STR, RANDOM_SEED, seed_everything, get_device
)

# ======================== 核心参数配置 ========================
# 4种GFP类型全做
ALL_GFP_TYPES = ['cgreGFP', 'ppluGFP', 'amacGFP', 'avGFP']

# 各类型训练数据最高亮度（共识目标）
TYPE_CONSENSUS_TARGET = {
    'cgreGFP': 4.55,   # 训练max=4.603
    'ppluGFP': 4.40,   # 训练max=4.445
    'amacGFP': 3.80,   # V7天花板~3.89, 下调
    'avGFP': 3.45,     # V7天花板~3.61, 下调
}

# V7初筛: 各类型WT亮度 - 0.1（筛掉明显差的）
V7_FILTER_PER_TYPE = {
    'cgreGFP': 4.40,   # WT=4.4969
    'ppluGFP': 4.13,   # WT=4.2258
    'amacGFP': 3.80,   # WT=3.9707, 下调(P95附近)
    'avGFP': 3.50,     # WT=3.7192, 下调(P95附近)
}

# 三模型权重（V9最精确, V7次之, V5最低）
MODEL_WEIGHTS = {
    'v5': 0.15,  # R²=0.73 但实际泛化弱, 权重最低
    'v7': 0.35,  # R²=0.71 Frozen MLP, 稳定可靠
    'v9': 0.50   # R²=0.89 LoRA, 最精确
}

# 共识条件
CONSENSUS_VOTES = 2  # 至少2个模型达到该类型共识目标

# 迭代参数
MAX_ITERATIONS = 5          # 5轮迭代, 每轮不同seed生成不同变体
TOP_N_PER_TYPE = 50         # 每类型Top 50父本

# 突变参数
N_VARIANTS_PER_PARENT = 500
N_MUTATIONS = 3
CHROMOPHORE_AND_CORE_INDICES = [
    64, 65, 66,  # 发色团 TYG (Thr-Tyr-Gly)
    61, 62, 63, 94, 96, 142, 144, 146, 148, 163, 165, 193, 211
]
ALL_AA = "ACDEFGHIKLMNPQRSTVWY"

GFP_TYPE_TO_IDX = {'avGFP': 0, 'amacGFP': 1, 'cgreGFP': 2, 'ppluGFP': 3}
GFP_WT_BRIGHTNESS = {
    'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258
}


# ======================== 突变解析 ========================
def parse_mutations(mut_str, wt_seq):
    """解析 'A109D:K130M' 格式的突变字符串, 返回完整序列"""
    if pd.isna(mut_str) or str(mut_str).strip().upper() in ["WT", ""]:
        return wt_seq
    seq_list = list(wt_seq)
    mutations = re.findall(r'([A-Za-z])(\d+)([A-Za-z*])', str(mut_str))
    if not mutations:
        return None
    for wt_aa, pos_str, mut_aa in mutations:
        pos = int(pos_str) - 1  # 1-based -> 0-based
        mut_aa = mut_aa.upper()
        if mut_aa == '*':
            return None
        if 0 <= pos < len(seq_list):
            seq_list[pos] = mut_aa
    return "".join(seq_list)


# ======================== Checkpoint 支持 ========================
CKPT_DIR = GEN_SEQS_DIR / "checkpoints"

def extract_esm2_embeddings_checkpointed(base_model, tokenizer, device, sequences,
                                        batch_size=64, chunk_size=10000, iteration=0):
    """带checkpoint的ESM-2嵌入提取, 每chunk_size条存盘一次, 断电后从断点继续"""
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"iter{iteration}_embeddings.npy"
    ckpt_meta = CKPT_DIR / f"iter{iteration}_meta.npz"

    n_seqs = len(sequences)
    start_chunk = 0

    # 尝试加载已有checkpoint
    if ckpt_meta.exists() and ckpt_path.exists():
        meta = np.load(ckpt_meta, allow_pickle=True)
        n_processed = int(meta['n_processed'])
        saved_n_seqs = int(meta['n_seqs'])
        if n_processed > 0 and saved_n_seqs == n_seqs:
            try:
                embeddings = np.load(ckpt_path)
                if embeddings.shape[0] == n_processed and embeddings.shape[1] == 1280:
                    start_chunk = n_processed // chunk_size
                    print(f"  Checkpoint恢复: 已处理{n_processed}/{n_seqs}条, "
                          f"从chunk {start_chunk}继续")
                    if n_processed >= n_seqs:
                        return embeddings[:n_seqs]
                else:
                    print(f"  Checkpoint形状不匹配, 重新开始")
                    n_processed = 0
            except Exception as e:
                print(f"  Checkpoint加载失败: {e}, 重新开始")
                n_processed = 0

    # 分配输出数组
    embeddings = np.zeros((n_seqs, 1280), dtype=np.float32)

    # 如有checkpoint, 填充已有数据
    if start_chunk > 0 and ckpt_path.exists():
        try:
            loaded = np.load(ckpt_path)
            embeddings[:loaded.shape[0]] = loaded
        except Exception:
            start_chunk = 0

    n_chunks = (n_seqs + chunk_size - 1) // chunk_size

    with torch.no_grad():
        for chunk_idx in range(start_chunk, n_chunks):
            cs = chunk_idx * chunk_size
            ce = min(cs + chunk_size, n_seqs)
            chunk_seqs = sequences[cs:ce]

            chunk_embs = []
            for i in tqdm(range(0, len(chunk_seqs), batch_size),
                         desc=f"  ESM-2 chunk {chunk_idx+1}/{n_chunks}",
                         leave=(chunk_idx == n_chunks - 1)):
                batch = chunk_seqs[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=250).to(device)
                outputs = base_model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                chunk_embs.append(pooled.cpu().numpy())

            chunk_embs_arr = np.concatenate(chunk_embs, axis=0)
            embeddings[cs:cs+len(chunk_embs_arr)] = chunk_embs_arr

            # 存盘
            np.save(ckpt_path, embeddings[:ce])
            np.savez(ckpt_meta, n_processed=ce, n_seqs=n_seqs)
            print(f"  Checkpoint保存: {ce}/{n_seqs}条 ({ce/n_seqs*100:.1f}%)")

    # 完成后删除checkpoint
    try:
        ckpt_path.unlink(missing_ok=True)
        ckpt_meta.unlink(missing_ok=True)
    except Exception:
        pass

    return embeddings


# ======================== 模型加载 (按需, 节省显存) ========================
# 显存策略: V7 MLP头极小(~10MB), 可与V9/V5同时加载
#   1. 加载ESM-2 base → 提取嵌入 → V7打分
#   2. 卸载ESM-2 base → 加载V9 LoRA → V9打分
#   3. 卸载V9 → 加载V5 LoRA → V5打分
#   全程V7 MLP头常驻显存

def clear_gpu_cache():
    """清理GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ResidualBrightnessMLP(nn.Module):
    """V7.5架构: 粗预测 + 残差修正 + 可学习基线"""
    def __init__(self, input_dim=1280, num_gfp_types=4):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        total_input = input_dim + num_gfp_types

        self.backbone = nn.Sequential(
            nn.Linear(total_input, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
        )

        self.coarse_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        self.residual_head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1)
        )
        for m in self.residual_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

        self.type_baseline = nn.Parameter(torch.tensor([3.7192, 3.9707, 4.4969, 4.2258]))

    def forward(self, embeddings, gfp_type_idx):
        B = embeddings.size(0)
        x = self.input_norm(embeddings)
        type_onehot = torch.zeros(B, 4, device=embeddings.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        x = torch.cat([x, type_onehot], dim=1)
        features = self.backbone(x)
        coarse_delta = self.coarse_head(features).squeeze(-1)
        residual = self.residual_head(features).squeeze(-1)
        delta_final = coarse_delta + residual
        baseline = self.type_baseline[gfp_type_idx]
        pred_abs = delta_final + baseline
        return coarse_delta, delta_final, pred_abs


def load_esm2_base_for_embeddings(device=None):
    """加载ESM-2 base模型用于提取嵌入"""
    if device is None:
        device = get_device()
    print("  加载ESM-2基础模型 (提取嵌入)...")
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    base_model = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR)).to(device)
    base_model.eval()
    print(f"  ESM-2加载完成 ({device})")
    return base_model, tokenizer, device


def extract_esm2_embeddings(base_model, tokenizer, device, sequences, batch_size=64):
    """用冻结ESM-2提取pooled嵌入"""
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="  ESM-2嵌入提取"):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=250).to(device)
            outputs = base_model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embeddings.append(pooled.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def load_v7_model():
    """加载V7 Frozen MLP的MLP头"""
    device = get_device()
    model_dirs = list(MODELS_DIR.glob("ESM2_ResidualMLP_V75_72k_*"))
    if not model_dirs:
        model_dirs = list(MODELS_DIR.glob("ESM2_Frozen_MLP_V7_72k_*"))
    if not model_dirs:
        raise FileNotFoundError("未找到V7/V7.5模型!")

    model_dir = sorted(model_dirs)[-1]
    print(f"  加载V7 MLP头: {model_dir.name}")

    config_path = model_dir / "model_config.joblib"
    try:
        config = joblib.load(config_path)
    except Exception:
        config = torch.load(config_path, map_location=device, weights_only=False)
    input_dim = config.get('input_dim', 1280)
    model = ResidualBrightnessMLP(input_dim=input_dim, num_gfp_types=4)
    pt_path = model_dir / "regressor.pt"
    if not pt_path.exists():
        pt_path = model_dir / "mlp_model.pt"
    model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model, device


def predict_v7(model, device, embeddings, gfp_type_idx, batch_size=256):
    """V7预测绝对亮度"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            emb_batch = torch.FloatTensor(embeddings[i:i+batch_size]).to(device)
            type_batch = torch.LongTensor(gfp_type_idx[i:i+batch_size]).to(device)
            _, _, pred_abs = model(emb_batch, type_batch)
            all_preds.append(pred_abs.cpu().numpy())
    return np.concatenate(all_preds)


# ======================== V5 LoRA 模型 ========================
class ESM2LoRARegressorV5(nn.Module):
    def __init__(self, base_model, num_gfp_types=4):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        input_dim = hidden_size + num_gfp_types
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 768), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.num_gfp_types = num_gfp_types

    def forward(self, input_ids, attention_mask, gfp_type_idx):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        type_onehot = torch.zeros(pooled.size(0), self.num_gfp_types, device=pooled.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        combined = torch.cat([pooled, type_onehot], dim=1)
        return self.regressor(combined).squeeze(-1)


def load_v5_model():
    """加载V5 LoRA模型"""
    device = get_device()
    model_dirs = list(MODELS_DIR.glob("ESM2_LoRA_V5_TypeEmbed_*"))
    if not model_dirs:
        raise FileNotFoundError("未找到V5 LoRA模型!")
    model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
    print(f"  加载V5 LoRA: {model_dir.name}")
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    base = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR)).to(device)
    base.eval()
    peft = PeftModel.from_pretrained(base, str(model_dir))
    model = ESM2LoRARegressorV5(peft, num_gfp_types=4).to(device)
    model.regressor.load_state_dict(
        torch.load(model_dir / "regressor.pt", map_location=device, weights_only=False)
    )
    model.eval()
    return tokenizer, model, device


def predict_v5(tokenizer, model, device, sequences, gfp_type_idx, batch_size=64):
    """V5预测delta值"""
    all_deltas = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="V5预测"):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=250).to(device)
            type_batch = torch.LongTensor(gfp_type_idx[i:i+batch_size]).to(device)
            deltas = model(inputs['input_ids'], inputs['attention_mask'], type_batch)
            all_deltas.append(deltas.cpu().float().numpy())
    return np.concatenate(all_deltas)


# ======================== V9 LoRA 模型 ========================
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

    def forward(self, input_ids, attention_mask, gfp_type_idx):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.pool_norm(pooled)
        type_onehot = torch.zeros(pooled.size(0), self.num_gfp_types, device=pooled.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        combined = torch.cat([pooled, type_onehot], dim=1)
        return self.regressor(combined).squeeze(-1)


def load_v9_model():
    """加载V9 LoRA模型"""
    device = get_device()
    model_dirs = list(MODELS_DIR.glob("ESM2_LoRA_V9_72k_*"))
    if not model_dirs:
        raise FileNotFoundError("未找到V9 LoRA模型!")
    model_dir = sorted(model_dirs)[-1]
    print(f"  加载V9 LoRA: {model_dir.name}")
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    base_model = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR))
    lora_model = PeftModel.from_pretrained(base_model, model_dir)
    lora_model = lora_model.merge_and_unload()
    regressor = ESM2LoRARegressorV9(lora_model)
    head_state = torch.load(model_dir / "regressor.pt", map_location=device, weights_only=False)
    first_key = list(head_state.keys())[0]
    if first_key.startswith("regressor.") or first_key.startswith("pool_norm."):
        regressor_state = {k.replace("regressor.", ""): v for k, v in head_state.items() if k.startswith("regressor.")}
        regressor.regressor.load_state_dict(regressor_state)
    else:
        regressor.regressor.load_state_dict(head_state)
    regressor.to(device)
    regressor.eval()
    return tokenizer, regressor, device


def predict_v9(tokenizer, model, device, sequences, gfp_type_idx, batch_size=32):
    """V9预测delta值"""
    all_deltas = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="V9预测"):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=250).to(device)
            type_batch = torch.LongTensor(gfp_type_idx[i:i+batch_size]).to(device)
            deltas = model(inputs['input_ids'], inputs['attention_mask'], type_batch)
            all_deltas.append(deltas.cpu().float().numpy())
    return np.concatenate(all_deltas)


# ======================== 突变生成 ========================
def generate_variants(parent_df, seed=RANDOM_SEED):
    """对每条父本生成N_VARIANTS_PER_PARENT个突变变体"""
    mutable_indices = [i for i in range(len(WT_AVGFP)) if i not in CHROMOPHORE_AND_CORE_INDICES]
    rng = np.random.default_rng(seed)
    all_variants = []

    for idx, row in tqdm(parent_df.iterrows(), total=len(parent_df), desc="生成变体"):
        parent_seq = row['sequence']
        parent_list = list(parent_seq)
        gfp_type = row['gfp_type']
        type_idx = row['type_idx']

        for _ in range(N_VARIANTS_PER_PARENT):
            seq_copy = parent_list.copy()
            n_mut = min(N_MUTATIONS, len(mutable_indices))
            positions = rng.choice(mutable_indices, n_mut, replace=False)
            for pos in positions:
                current_aa = seq_copy[pos]
                alternatives = [aa for aa in ALL_AA if aa != current_aa]
                seq_copy[pos] = rng.choice(alternatives)
            all_variants.append({
                'sequence': "".join(seq_copy),
                'parent_idx': idx,
                'n_mutations': len(positions),
                'parent_rank': row['rank'],
                'parent_brightness': row['brightness'],
                'gfp_type': gfp_type,
                'type_idx': type_idx,
            })

    variant_df = pd.DataFrame(all_variants)
    return variant_df


# ======================== 主流程 ========================
def generate_phase2_hierarchical(force=False, max_iterations=None, start_iter=1):
    """Phase 2: 分层突变策略 - 4类型独立阈值, 多轮迭代"""
    seed_everything(RANDOM_SEED)
    if max_iterations is not None:
        global MAX_ITERATIONS
        MAX_ITERATIONS = max_iterations

    out_path = GEN_SEQS_DIR / f"phase2_hierarchical_{AUTHOR}_{DATE_STR}.csv"
    if out_path.exists() and not force:
        print(f"已存在: {out_path}")
        print(f"  如需重跑: python src/phase2_generate_hierarchical.py --force")
        return out_path

    print("=" * 60)
    print(" Phase 2: 分层突变策略 - 4类型独立阈值")
    print("=" * 60)
    for gt in ALL_GFP_TYPES:
        print(f"  {gt}: 共识目标={TYPE_CONSENSUS_TARGET[gt]}, V7初筛>={V7_FILTER_PER_TYPE[gt]}")
    print(f"  共识票数: >= {CONSENSUS_VOTES}/3")
    print(f"  权重: V5={MODEL_WEIGHTS['v5']:.0%}, V7={MODEL_WEIGHTS['v7']:.0%}, V9={MODEL_WEIGHTS['v9']:.0%}")

    print(f"  迭代: {MAX_ITERATIONS}轮")
    print(f"  父本: 每类型Top {TOP_N_PER_TYPE}")

    # ========== Step 1: 加载各类型Top N父本 ==========
    print(f"\n{'='*60}")
    print(f" Step 1: 加载各类型Top {TOP_N_PER_TYPE}父本")
    print(f"{'='*60}")

    raw_df = pd.read_excel(RAW_DIR / "GFP_data.xlsx", sheet_name="brightness")
    parent_records = []

    for gfp_type in ALL_GFP_TYPES:
        type_idx = GFP_TYPE_TO_IDX[gfp_type]
        type_df = raw_df[raw_df['GFP type'].str.contains(gfp_type, case=False, na=False)].copy()
        print(f"\n  {gfp_type}: 共 {len(type_df)} 条")

        if len(type_df) == 0:
            continue

        type_df = type_df.sort_values('Brightness', ascending=False)
        top_df = type_df.head(TOP_N_PER_TYPE)

        print(f"    Top {TOP_N_PER_TYPE} (亮度 {top_df['Brightness'].min():.4f}~{top_df['Brightness'].max():.4f}):")
        for i, (_, row) in enumerate(top_df.head(5).iterrows(), 1):
            mut_str = str(row['aaMutations']) if pd.notna(row['aaMutations']) else 'WT'
            print(f"      #{i}: brightness={row['Brightness']:.4f}, mutations={mut_str}")

        for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
            full_seq = parse_mutations(row['aaMutations'], WT_AVGFP)
            if full_seq and len(full_seq) == len(WT_AVGFP):
                parent_records.append({
                    'sequence': full_seq,
                    'brightness': row['Brightness'],
                    'gfp_type': gfp_type,
                    'type_idx': type_idx,
                    'rank': rank,
                    'mutations': str(row['aaMutations'])
                })

    parent_df = pd.DataFrame(parent_records)
    print(f"\n  总计: {len(parent_df)} 条父本")
    for gt in ALL_GFP_TYPES:
        sub = parent_df[parent_df['gfp_type'] == gt]
        print(f"    {gt}: {len(sub)} 条")

    # ========== Step 2: 加载V7 MLP头(常驻显存) + ESM-2 ==========
    print(f"\n{'='*60}")
    print(" Step 2: 加载V7 MLP头 + ESM-2")
    print(f"{'='*60}")
    v7_model, v7_device = load_v7_model()
    esm2_base, esm2_tok, esm2_dev = load_esm2_base_for_embeddings()

    # ========== 多轮迭代: 生成变体 + V7初筛 ==========
    all_v7_passed = []
    seen_seqs = set()

    # 恢复: 加载已完成的迭代结果
    if start_iter > 1:
        print(f"\n  恢复: 从迭代 {start_iter} 开始, 加载前 {start_iter-1} 轮结果...")
        for prev_iter in range(1, start_iter):
            # 搜索所有日期的迭代文件 (防止跨日DATE_STR变化)
            pattern = f"phase2_iter{prev_iter}_{AUTHOR}_*.csv"
            matches = sorted(GEN_SEQS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                iter_file = matches[0]
                prev_df = pd.read_csv(iter_file)
                all_v7_passed.append(prev_df)
                seen_seqs.update(prev_df['sequence'].tolist())
                print(f"    迭代{prev_iter}: 加载 {len(prev_df)} 条 ({iter_file.name})")
            else:
                print(f"    迭代{prev_iter}: 文件不存在, 跳过")
        if all_v7_passed:
            total_prev = sum(len(d) for d in all_v7_passed)
            print(f"  已加载: {total_prev} 条, {len(seen_seqs)} 个不重复序列")

    for iteration in range(1, MAX_ITERATIONS + 1):
        if iteration < start_iter:
            continue
        print(f"\n{'#'*60}")
        print(f" 迭代 {iteration}/{MAX_ITERATIONS}")
        print(f"{'#'*60}")

        iter_seed = RANDOM_SEED + iteration * 1000
        variant_df = generate_variants(parent_df, seed=iter_seed)
        n_gen = len(variant_df)
        variant_df = variant_df[~variant_df['sequence'].isin(seen_seqs)]
        variant_df = variant_df.drop_duplicates(subset=['sequence']).reset_index(drop=True)
        n_new = len(variant_df)
        seen_seqs.update(variant_df['sequence'].tolist())
        print(f"  生成: {n_gen} 条, 新增(去重): {n_new} 条")

        if n_new == 0:
            print("  无新变体, 跳过")
            continue

        # V7预测 (带checkpoint, 断电后从断点继续)
        embeddings = extract_esm2_embeddings_checkpointed(
            esm2_base, esm2_tok, esm2_dev,
            variant_df['sequence'].tolist(),
            iteration=iteration
        )
        variant_df['pred_v7'] = predict_v7(v7_model, v7_device, embeddings, variant_df['type_idx'].values)

        # 按类型V7初筛
        v7_mask = np.zeros(n_new, dtype=bool)
        for gt in ALL_GFP_TYPES:
            type_mask = variant_df['gfp_type'].values == gt
            threshold = V7_FILTER_PER_TYPE[gt]
            v7_mask |= (type_mask & (variant_df['pred_v7'].values >= threshold))

        passed_df = variant_df[v7_mask].copy()

        print(f"  V7通过:")
        for gt in ALL_GFP_TYPES:
            n_b = (variant_df['gfp_type'] == gt).sum()
            n_a = (passed_df['gfp_type'] == gt).sum()
            if n_b > 0:
                print(f"    {gt}: {n_b}→{n_a} ({n_a/n_b*100:.1f}%)")
        print(f"  总计: {n_new}→{len(passed_df)} ({len(passed_df)/max(n_new,1)*100:.1f}%)")

        all_v7_passed.append(passed_df)

        # 保存本迭代中间结果
        if len(passed_df) > 0:
            iter_out = GEN_SEQS_DIR / f"phase2_iter{iteration}_{AUTHOR}_{DATE_STR}.csv"
            save_cols = [c for c in ['sequence', 'gfp_type', 'n_mutations', 'parent_rank',
                                     'parent_brightness', 'pred_v7'] if c in passed_df.columns]
            passed_df.sort_values('pred_v7', ascending=False)[save_cols].to_csv(iter_out, index=False)
            print(f"  已保存: {iter_out.name}")

        total_pool = sum(len(d) for d in all_v7_passed)
        print(f"  累计V7通过池: {total_pool} 条")

    # 释放ESM-2和V7
    del esm2_base, esm2_tok, v7_model
    clear_gpu_cache()

    # ========== 合并所有迭代的V7通过序列 ==========
    if not all_v7_passed:
        print("\n  所有迭代均无序列通过V7初筛!")
        return None

    stage1_df = pd.concat(all_v7_passed, ignore_index=True)
    stage1_df = stage1_df.sort_values('pred_v7', ascending=False).drop_duplicates(
        subset=['sequence'], keep='first').reset_index(drop=True)
    print(f"\n  累计去重后: {len(stage1_df)} 条V7通过序列")
    for gt in ALL_GFP_TYPES:
        n = (stage1_df['gfp_type'] == gt).sum()
        print(f"    {gt}: {n} 条")

    # 保存合并V7中间结果
    v7_out = GEN_SEQS_DIR / f"phase2_v7_filtered_{AUTHOR}_{DATE_STR}.csv"
    save_cols = [c for c in ['sequence', 'gfp_type', 'n_mutations', 'parent_rank',
                             'parent_brightness', 'pred_v7'] if c in stage1_df.columns]
    stage1_df.sort_values('pred_v7', ascending=False).head(2000)[save_cols].to_csv(v7_out, index=False)
    print(f"  V7合并结果: {v7_out.name}")

    # ========== Stage 2a: V9精筛 ==========
    print(f"\n{'='*60}")
    print(f" Stage 2a: V9 LoRA精筛 ({len(stage1_df)} 条)")
    print(f"{'='*60}")

    if len(stage1_df) == 0:
        print("  无序列可打分!")
        return None

    v9_tokenizer, v9_model, v9_device = load_v9_model()
    stage1_seqs = stage1_df['sequence'].tolist()
    # 从gfp_type重建type_idx (CSV加载的数据没有type_idx列)
    stage1_type_idx = stage1_df['gfp_type'].map(GFP_TYPE_TO_IDX).values
    if np.isnan(stage1_type_idx).any():
        raise ValueError(f"type_idx存在NaN, 未知gfp_type: {stage1_df['gfp_type'].unique()}")

    pred_v9_delta = predict_v9(v9_tokenizer, v9_model, v9_device, stage1_seqs, stage1_type_idx)
    pred_v9_abs = np.array([
        d + GFP_WT_BRIGHTNESS[t] for d, t in zip(pred_v9_delta, stage1_df['gfp_type'].values)
    ])
    stage1_df['pred_v9_delta'] = pred_v9_delta
    stage1_df['pred_v9_abs'] = pred_v9_abs

    del v9_tokenizer, v9_model
    clear_gpu_cache()
    print(f"  V9已卸载")

    # ========== Stage 2b: V5精筛 ==========
    print(f"\n{'='*60}")
    print(f" Stage 2b: V5 LoRA精筛 ({len(stage1_df)} 条)")
    print(f"{'='*60}")

    v5_tokenizer, v5_model, v5_device = load_v5_model()
    pred_v5_delta = predict_v5(v5_tokenizer, v5_model, v5_device, stage1_seqs, stage1_type_idx)
    pred_v5_abs = np.array([
        d + GFP_WT_BRIGHTNESS[t] for d, t in zip(pred_v5_delta, stage1_df['gfp_type'].values)
    ])
    stage1_df['pred_v5_delta'] = pred_v5_delta
    stage1_df['pred_v5_abs'] = pred_v5_abs

    del v5_tokenizer, v5_model
    clear_gpu_cache()
    print(f"  V5已卸载")

    # ========== Stage 3: 三模型共识筛选(按类型) ==========
    print(f"\n{'='*60}")
    print(f" Stage 3: 三模型共识筛选 (按类型独立阈值)")
    print(f"{'='*60}")

    w = MODEL_WEIGHTS
    stage1_df['weighted_avg'] = (
        w['v5'] * stage1_df['pred_v5_abs'] +
        w['v7'] * stage1_df['pred_v7'] +
        w['v9'] * stage1_df['pred_v9_abs']
    )

    # 按类型计算共识票数
    stage1_df['votes'] = 0
    stage1_df['type_target'] = 0.0
    for gt in ALL_GFP_TYPES:
        target = TYPE_CONSENSUS_TARGET[gt]
        type_mask = stage1_df['gfp_type'].values == gt
        stage1_df.loc[type_mask, 'type_target'] = target
        stage1_df.loc[type_mask, 'votes'] = (
            (stage1_df.loc[type_mask, 'pred_v5_abs'].values >= target).astype(int) +
            (stage1_df.loc[type_mask, 'pred_v7'].values >= target).astype(int) +
            (stage1_df.loc[type_mask, 'pred_v9_abs'].values >= target).astype(int)
        )

    # 共识筛选
    consensus_df = stage1_df[stage1_df['votes'] >= CONSENSUS_VOTES].copy()
    consensus_df = consensus_df.sort_values('weighted_avg', ascending=False)

    # ========== 结果汇报 ==========
    print(f"\n{'='*60}")
    print(" 结果汇报")
    print(f"{'='*60}")

    for gt in ALL_GFP_TYPES:
        type_stage1 = stage1_df[stage1_df['gfp_type'] == gt]
        type_consensus = consensus_df[consensus_df['gfp_type'] == gt]
        target = TYPE_CONSENSUS_TARGET[gt]
        wt = GFP_WT_BRIGHTNESS[gt]

        print(f"\n  {gt} (WT={wt:.3f}, 目标={target}):")
        print(f"    V7初筛后(累计{MAX_ITERATIONS}轮): {len(type_stage1)} 条")
        if len(type_stage1) > 0:
            print(f"    V7范围: [{type_stage1['pred_v7'].min():.3f}, {type_stage1['pred_v7'].max():.3f}]")
            print(f"    V9范围: [{type_stage1['pred_v9_abs'].min():.3f}, {type_stage1['pred_v9_abs'].max():.3f}]")
            print(f"    加权平均: [{type_stage1['weighted_avg'].min():.3f}, {type_stage1['weighted_avg'].max():.3f}]")
        print(f"    共识通过: {len(type_consensus)} 条")

        if len(type_consensus) > 0:
            best = type_consensus.iloc[0]
            print(f"    最佳: 加权={best['weighted_avg']:.3f}, V5={best['pred_v5_abs']:.3f}, V7={best['pred_v7']:.3f}, V9={best['pred_v9_abs']:.3f}")

    # ========== 保存最终结果 ==========
    print(f"\n{'='*60}")
    print(" 保存最终结果")
    print(f"{'='*60}")

    output_cols = [
        'sequence', 'gfp_type', 'n_mutations', 'parent_rank',
        'parent_brightness', 'pred_v5_abs', 'pred_v7', 'pred_v9_abs',
        'pred_v5_delta', 'pred_v9_delta',
        'weighted_avg', 'votes', 'type_target'
    ]
    final_output = consensus_df[output_cols].copy()
    final_output.columns = [
        'sequence', 'gfp_type', 'num_mutations', 'parent_rank',
        'parent_brightness', 'pred_v5', 'pred_v7', 'pred_v9',
        'pred_v5_delta', 'pred_v9_delta',
        'weighted_avg', 'consensus_votes', 'type_consensus_target'
    ]
    final_output.to_csv(out_path, index=False)

    print(f"\n  最终结果: {len(final_output)} 条共识序列")
    print(f"  保存至: {out_path}")

    # 各类型Top 5
    print(f"\n  各类型Top 5:")
    for gt in ALL_GFP_TYPES:
        type_top = final_output[final_output['gfp_type'] == gt].head(5)
        if len(type_top) > 0:
            print(f"\n  {gt}:")
            for i, (_, row) in enumerate(type_top.iterrows(), 1):
                print(f"    {i}. 加权={row['weighted_avg']:.3f} V5={row['pred_v5']:.3f} V7={row['pred_v7']:.3f} V9={row['pred_v9']:.3f} 票数={row['consensus_votes']}")
        else:
            print(f"\n  {gt}: 无共识序列")

    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--start-iter", type=int, default=1)
    args = parser.parse_args()
    generate_phase2_hierarchical(force=args.force, max_iterations=args.iterations, start_iter=args.start_iter)
