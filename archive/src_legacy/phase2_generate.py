"""
Phase 2: 从高亮度已知序列出发 + 少量随机突变 + LoRA筛选
策略:
  1. 从训练数据取Top N最亮序列 (4种GFP各1000条, 已实验验证)
  2. 每条随机突变1~3个位点, 生成多个变体
  3. 用V5 LoRA打分 (用正确的GFP类型索引), 保留预测亮度 ≥ 父本的变体

存档产出:
    generated_seqs/phase2_bright_mut_{num}_{AUTHOR}_{DATE}.csv
    包含列: seq_id, sequence, source, gfp_type, num_mutations_from_parent,
            parent_brightness, pred_brightness, pred_delta_vs_parent, parent_mutations

队友同步:
    只需同步此 CSV 文件即可，无需 GPU。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from src.config import (
    GEN_SEQS_DIR, MODELS_DIR, LOCAL_ESM2_DIR, RAW_DIR,
    WT_AVGFP, WT_SFGFP, AUTHOR, DATE_STR, RANDOM_SEED, seed_everything, get_device
)

# sfGFP 发色团及核心刚性区域索引 (0-based), 绝对不允许修改
CHROMOPHORE_AND_CORE_INDICES = [
    64, 65, 66,  # 发色团 TYG (Thr-Tyr-Gly)
    61, 62, 63, 94, 96, 142, 144, 146, 148, 163, 165, 193, 211
]

ALL_AA = "ACDEFGHIKLMNPQRSTVWY"

# 4种GFP类型配置 (与V5训练一致)
GFP_TYPES = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {t: i for i, t in enumerate(GFP_TYPES)}
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
        pos = int(pos_str) - 1  # 1-based → 0-based
        mut_aa = mut_aa.upper()
        if mut_aa == '*':
            return None  # 禁止终止密码子
        if 0 <= pos < len(seq_list):
            seq_list[pos] = mut_aa
    return "".join(seq_list)


# ======================== V5 模型加载 ========================
class ESM2LoRARegressorV5(nn.Module):
    """与训练脚本 phase1_lora_finetune_v5.py 完全一致的模型结构"""
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


def load_lora_scorer():
    """加载V5 LoRA模型作为亮度打分器"""
    device = get_device()
    model_dirs = list(MODELS_DIR.glob("ESM2_LoRA_V5_TypeEmbed_*"))
    if not model_dirs:
        raise FileNotFoundError("❌ 未找到V5 LoRA模型! 请先运行 Phase 1 训练")
    model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
    print(f"-> 加载LoRA打分器: {model_dir.name}")

    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_ESM2_DIR))
    base = AutoModel.from_pretrained(str(LOCAL_ESM2_DIR)).to(device)
    base.eval()
    peft = PeftModel.from_pretrained(base, str(model_dir))
    model = ESM2LoRARegressorV5(peft, num_gfp_types=4).to(device)
    model.regressor.load_state_dict(
        torch.load(model_dir / "regressor.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return tokenizer, model, device


def predict_deltas(tokenizer, model, device, sequences, gfp_type_idx, batch_size=64):
    """批量预测delta亮度 (相对于avGFP WT), 支持指定GFP类型索引"""
    deltas = []
    type_tensor_single = torch.tensor([gfp_type_idx], dtype=torch.long, device=device)
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=250).to(device)
        type_tensor = type_tensor_single.expand(len(batch))
        with torch.no_grad(), torch.amp.autocast('cuda'):
            pred = model(inputs['input_ids'], inputs['attention_mask'], type_tensor)
            deltas.extend(pred.cpu().float().numpy())
    return np.array(deltas)


# ======================== 主函数 ========================
def generate_sequences(num_samples: int = 5000, mask_ratio: float = 0.15,
                       force: bool = False):
    """
    从高亮度已知序列出发 + 随机突变 + LoRA粗筛(删死蛋白)
    支持全部4种GFP类型, 每种各取top 1000
    num_samples: 每种GFP的父本数 (默认1000)
    """
    seed_everything()
    device = get_device()

    print("\n" + "=" * 60)
    print(f"🧬 [Phase 2] 高亮度序列 + 随机突变 + LoRA粗筛 (设备: {device})")
    print("=" * 60)

    out_path = GEN_SEQS_DIR / f"phase2_bright_mut_{num_samples}_{AUTHOR}_{DATE_STR}.csv"
    if out_path.exists() and not force:
        print(f"⚠️ 已存在存档: {out_path.name}")
        print("   跳过生成。强制重跑: python main.py generate --force")
        return out_path

    # ========== Step 1: 加载各GFP类型的高亮度序列 ==========
    print(f"\n{'='*40}")
    print("📊 Step 1: 加载高亮度序列 (4种GFP)")
    print(f"{'='*40}")

    raw_df = pd.read_excel(RAW_DIR / "GFP_data.xlsx")

    top_n_per_type = 1000   # 每种GFP取1000条父本
    variants_per_seq = 20   # 每条父本生成20个变体
    max_muts = 5            # 突变范围 1~5 个
    dead_threshold = -0.3   # 亮度下降超过0.3视为死蛋白, 删除

    # 收集所有父本序列 (包含类型信息)
    parent_seqs = []      # 完整序列
    parent_brightness = [] # 实验亮度
    parent_mutations = []  # 突变字符串
    parent_type_idx = []   # GFP类型索引

    for gfp_type in GFP_TYPES:
        type_idx = GFP_TYPE_TO_IDX[gfp_type]
        type_df = raw_df[raw_df['GFP type'].str.contains(gfp_type, case=False, na=False)].copy()
        print(f"\n  {gfp_type}: 共 {len(type_df)} 条")

        if len(type_df) == 0:
            continue

        top_df = type_df.nlargest(top_n_per_type, 'Brightness')
        print(f"    选取Top {len(top_df)} (亮度 {top_df['Brightness'].min():.3f}~{top_df['Brightness'].max():.3f})")

        for _, row in top_df.iterrows():
            full_seq = parse_mutations(row['aaMutations'], WT_AVGFP)
            if full_seq and len(full_seq) == len(WT_AVGFP):
                parent_seqs.append(full_seq)
                parent_brightness.append(row['Brightness'])
                parent_mutations.append(str(row['aaMutations']))
                parent_type_idx.append(type_idx)

    print(f"\n  总计解析: {len(parent_seqs)} 条父本序列")
    parent_brightness = np.array(parent_brightness)
    parent_type_idx = np.array(parent_type_idx)

    # ========== Step 2: 生成随机突变变体 ==========
    print(f"\n{'='*40}")
    print(f"🧪 Step 2: 生成随机突变变体 (每条 {variants_per_seq} 个, 1~{max_muts}突变)")
    print(f"{'='*40}")

    mutable_indices = [i for i in range(len(WT_AVGFP)) if i not in CHROMOPHORE_AND_CORE_INDICES]
    rng = np.random.default_rng(RANDOM_SEED)

    all_variants = []  # (variant_seq, parent_idx, n_muts)
    for idx, parent_seq in enumerate(parent_seqs):
        parent_list = list(parent_seq)
        for _ in range(variants_per_seq):
            n_muts = rng.integers(1, max_muts + 1)
            seq_copy = parent_list.copy()
            positions = rng.choice(mutable_indices, min(n_muts, len(mutable_indices)), replace=False)
            for pos in positions:
                current_aa = seq_copy[pos]
                alternatives = [aa for aa in ALL_AA if aa != current_aa]
                seq_copy[pos] = rng.choice(alternatives)
            all_variants.append(("".join(seq_copy), idx, len(positions)))

    print(f"  生成变体: {len(all_variants)} 个")

    # ========== Step 3: LoRA打分 (按类型分别打分) ==========
    print(f"\n{'='*40}")
    print(f"📡 Step 3: LoRA打分 (按GFP类型)")
    print(f"{'='*40}")

    tokenizer, model, device = load_lora_scorer()

    # 打分父本 (按类型分组)
    print(f"  打分父本序列 ({len(parent_seqs)} 条)...")
    parent_deltas = np.zeros(len(parent_seqs))
    for type_idx in range(len(GFP_TYPES)):
        mask = parent_type_idx == type_idx
        if mask.sum() == 0:
            continue
        seqs = [parent_seqs[i] for i in range(len(parent_seqs)) if mask[i]]
        deltas = predict_deltas(tokenizer, model, device, seqs, type_idx, batch_size=64)
        parent_deltas[mask] = deltas
        print(f"    {GFP_TYPES[type_idx]}: {mask.sum()} 条")

    # 打分变体 (按父本类型分组)
    variant_seqs = [v[0] for v in all_variants]
    variant_type_idx = np.array([parent_type_idx[v[1]] for v in all_variants])
    print(f"  打分变体序列 ({len(variant_seqs)} 条)...")
    variant_deltas = np.zeros(len(variant_seqs))
    for type_idx in range(len(GFP_TYPES)):
        mask = variant_type_idx == type_idx
        if mask.sum() == 0:
            continue
        seqs = [variant_seqs[i] for i in range(len(variant_seqs)) if mask[i]]
        deltas = predict_deltas(tokenizer, model, device, seqs, type_idx, batch_size=64)
        variant_deltas[mask] = deltas

    # ========== Step 4: 筛选 ==========
    print(f"\n{'='*40}")
    print(f"✅ Step 4: 筛选结果")
    print(f"{'='*40}")

    all_candidates = []
    kept = 0
    total = 0

    for (var_seq, parent_idx, n_muts), var_delta in zip(all_variants, variant_deltas):
        total += 1
        parent_delta = parent_deltas[parent_idx]
        improvement = float(var_delta) - float(parent_delta)

        if improvement >= dead_threshold:
            kept += 1
            gfp_type = GFP_TYPES[parent_type_idx[parent_idx]]
            wt_bright = GFP_WT_BRIGHTNESS[gfp_type]
            pred_brightness = float(var_delta) + wt_bright
            all_candidates.append({
                'sequence': var_seq,
                'source': 'bright_mut',
                'gfp_type': gfp_type,
                'num_mutations_from_parent': n_muts,
                'parent_brightness': parent_brightness[parent_idx],
                'pred_brightness': pred_brightness,
                'pred_delta_vs_parent': improvement,
                'parent_mutations': parent_mutations[parent_idx],
                'parent_sequence': parent_seqs[parent_idx]
            })

    # 加入父本序列作为对照
    for idx, (seq, bright, muts, tidx) in enumerate(
            zip(parent_seqs, parent_brightness, parent_mutations, parent_type_idx)):
        gfp_type = GFP_TYPES[tidx]
        wt_bright = GFP_WT_BRIGHTNESS[gfp_type]
        pred_bright = float(parent_deltas[idx]) + wt_bright
        all_candidates.append({
            'sequence': seq,
            'source': 'parent',
            'gfp_type': gfp_type,
            'num_mutations_from_parent': 0,
            'parent_brightness': bright,
            'pred_brightness': pred_bright,
            'pred_delta_vs_parent': 0.0,
            'parent_mutations': muts,
            'parent_sequence': seq
        })

    # 排序: 按pred_brightness降序
    all_candidates.sort(key=lambda x: x['pred_brightness'], reverse=True)

    # 构建输出DataFrame
    df_out = pd.DataFrame(all_candidates)
    df_out.insert(0, 'seq_id', [f"bright_{i:05d}" for i in range(len(df_out))])

    # 去重
    before_dedup = len(df_out)
    df_out = df_out.drop_duplicates(subset=['sequence']).reset_index(drop=True)
    df_out['seq_id'] = [f"bright_{i:05d}" for i in range(len(df_out))]
    df_out.to_csv(out_path, index=False)

    # 统计
    n_mutants = sum(df_out['source'] == 'bright_mut')
    n_parents = sum(df_out['source'] == 'parent')
    print(f"\n{'='*60}")
    print(f"✅ 序列生成完成!")
    print(f"   父本序列: {n_parents} 条")
    for gfp_type in GFP_TYPES:
        n = sum(df_out['gfp_type'] == gfp_type)
        print(f"     {gfp_type}: {n} 条")
    print(f"   突变变体: {total} 条 → 保留 (≥父本{dead_threshold:+.1f}): {kept} 条 ({kept/max(total,1)*100:.1f}%)")
    print(f"   去重后: {len(df_out)} 条 (突变{n_mutants} + 父本{n_parents})")
    if n_mutants > 0:
        best = df_out.iloc[0]
        print(f"   🏆 最佳: 预测亮度={best['pred_brightness']:.4f}, "
              f"父本亮度={best['parent_brightness']:.4f}, "
              f"改善={best['pred_delta_vs_parent']:+.4f}, "
              f"类型={best['gfp_type']}")
        print(f"      父本突变: {best['parent_mutations']}")
    print(f"💾 存档: {out_path}")
    print(f"💡 队友只需同步此 CSV，无需 GPU")
    return out_path
