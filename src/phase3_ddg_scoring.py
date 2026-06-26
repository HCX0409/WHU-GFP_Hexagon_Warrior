"""
Phase 3: ESM3-ddG 热稳定性打分
对 Phase 2 产出的候选序列，使用 ESM3-ddG 模型预测每个突变的稳定性变化 (ddG)。
ddG < 0 = 稳定化突变 (好), ddG > 0 = 去稳定化突变 (坏)

输入:
    generated_seqs/phase2_bright_mut_{num}_{AUTHOR}_{DATE}.csv (Phase 2 产出)

存档产出:
    generated_seqs/phase3_ddg_scored_{num}_{AUTHOR}_{DATE}.csv
    包含列: seq_id, sequence, source, gfp_type, parent_brightness, pred_brightness,
            pred_delta_vs_parent, total_ddg, num_mutations_from_parent, parent_mutations

队友同步:
    需要: Phase 2 CSV + esm3_ddg_v2 模型权重
    需要 GPU (ESM3 模型较大, ~5.6GB)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')  # 强制离线, 避免网络请求
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from src.config import (
    GEN_SEQS_DIR, PRETRAINED_MODELS_DIR, LOCAL_ESM3_DDG_DIR,
    WT_AVGFP, AUTHOR, DATE_STR, RANDOM_SEED, seed_everything, get_device
)
from src.phase2_generate import parse_mutations


# ======================== 模型加载 ========================
def load_ddg_model():
    """加载 ESM3-ddG 热稳定性预测模型"""
    device = get_device()

    if LOCAL_ESM3_DDG_DIR.exists():
        model_path = str(LOCAL_ESM3_DDG_DIR)
        print(f"-> 加载本地 ESM3-ddG: {model_path}")
    else:
        model_path = "hazemessam/esm3_ddg_v2"
        print(f"-> 下载 ESM3-ddG: {model_path} (首次运行, ~5.6GB)")

    # ESM3 使用自定义 tokenizer
    try:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    except ImportError:
        raise ImportError("❌ 请先安装 esm 包: pip install esm")

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()
    tokenizer = EsmSequenceTokenizer()

    return tokenizer, model, device


def tokenize_seq(tokenizer, seq, device):
    """将氨基酸序列转为 ESM3 token tensor"""
    # EsmSequenceTokenizer.encode 接受字符串 (不是列表)
    encoded = tokenizer.encode(seq)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    return input_ids


def predict_ddg_single(tokenizer, model, device, wt_seq, mut_seq, positions):
    """
    预测一组突变的 ddG (稳定性变化)
    positions: 突变位置列表 (0-based)
    ESM3-ddG 只支持单突变输入, 多突变需逐个调用后求和
    返回: total_ddg (float), 负值=更稳定
    """
    wt_ids = tokenize_seq(tokenizer, wt_seq, device)
    mut_ids = tokenize_seq(tokenizer, mut_seq, device)

    total_ddg = 0.0
    for pos in positions:
        pos_tensor = torch.tensor([pos], dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            output = model(wt_ids, mut_ids, positions=pos_tensor)
            logits = output['logits']
        total_ddg += float(logits.sum().cpu())

    return total_ddg


# ======================== 主函数 ========================
def score_stability(force: bool = False):
    """
    对 Phase 2 候选序列进行 ESM3-ddG 热稳定性打分
    """
    seed_everything()
    device = get_device()

    print("\n" + "=" * 60)
    print(f"🧬 [Phase 3] ESM3-ddG 热稳定性打分 (设备: {device})")
    print("=" * 60)

    # 1. 查找 Phase 2 产出 (最新的)
    phase2_files = sorted(GEN_SEQS_DIR.glob("phase2_bright_mut_*_HCX_*.csv"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    if not phase2_files:
        raise FileNotFoundError("❌ 未找到 Phase 2 产出! 请先运行: python main.py generate")

    phase2_path = phase2_files[0]
    print(f"-> 读取 Phase 2 存档: {phase2_path.name}")

    df = pd.read_csv(phase2_path)
    print(f"   总序列: {len(df)} 条")

    # 只处理突变变体 (source='bright_mut'), 父本不需要打分
    mutants_df = df[df['source'] == 'bright_mut'].copy()
    parents_df = df[df['source'] == 'parent'].copy()
    print(f"   突变变体: {len(mutants_df)} 条 (需要ddG打分)")
    print(f"   父本序列: {len(parents_df)} 条 (ddG=0, 直接保留)")

    if len(mutants_df) == 0:
        print("⚠️ 没有突变变体需要打分!")
        return None

    # 2. 断点续跑检测
    out_path = GEN_SEQS_DIR / f"phase3_ddg_scored_{AUTHOR}_{DATE_STR}.csv"
    if out_path.exists() and not force:
        print(f"⚠️ 已存在存档: {out_path.name}")
        print("   跳过打分。强制重跑: python main.py score_ddg --force")
        return out_path

    # 3. 加载 ESM3-ddG 模型
    tokenizer, model, device = load_ddg_model()

    # 4. 逐条打分
    # 对于每个突变变体:
    #   - 从 parent_mutations 重建父本序列
    #   - 比较父本 vs 突变体找到 1~3 个突变位置
    #   - ESM3-ddG 预测这些位置的 ddG
    print(f"\n{'='*40}")
    print(f"📡 开始ddG打分 ({len(mutants_df)} 条变体)")
    print(f"{'='*40}")

    ddg_scores = []
    errors = 0

    for idx, row in tqdm(mutants_df.iterrows(), total=len(mutants_df), desc="ddG打分"):
        mut_seq = row['sequence']

        # 优先使用 parent_sequence 列 (Phase 2 已存储)
        parent_seq = row.get('parent_sequence', None)
        if pd.isna(parent_seq) or not parent_seq:
            # 回退: 从 parent_mutations 重建
            parent_mut_str = str(row.get('parent_mutations', ''))
            try:
                parent_seq = parse_mutations(parent_mut_str, WT_AVGFP)
            except Exception:
                ddg_scores.append(0.0)
                errors += 1
                if errors <= 5:
                    print(f"  ⚠️ 重建父本失败 (seq {idx})")
                continue

        if not parent_seq or len(parent_seq) != len(mut_seq):
            ddg_scores.append(0.0)
            errors += 1
            continue

        # 比较父本 vs 突变体, 找到实际突变位置 (应该只有 1~5 个)
        mut_positions = [i for i, (a, b) in enumerate(zip(parent_seq, mut_seq)) if a != b]

        if len(mut_positions) == 0:
            ddg_scores.append(0.0)  # 无突变 (和父本一样)
            continue

        if len(mut_positions) > 10:
            # 太多突变位置说明父本重建有问题, 跳过
            ddg_scores.append(0.0)
            errors += 1
            if errors <= 5:
                print(f"  ⚠️ 突变位置过多 ({len(mut_positions)}) (seq {idx})")
            continue

        try:
            # 用父本序列作为 reference, 预测额外突变的 ddG
            total_ddg = predict_ddg_single(
                tokenizer, model, device,
                parent_seq, mut_seq, mut_positions
            )
            ddg_scores.append(total_ddg)
        except Exception as e:
            ddg_scores.append(0.0)
            errors += 1
            if errors <= 5:
                print(f"  ⚠️ 打分失败 (seq {idx}): {e}")

    mutants_df['total_ddg'] = ddg_scores
    print(f"\n  打分完成! 错误: {errors}/{len(mutants_df)}")

    # 5. 合并结果
    # 父本序列 ddG = 0
    parents_df['total_ddg'] = 0.0

    # 确保列一致
    for col in mutants_df.columns:
        if col not in parents_df.columns:
            parents_df[col] = None
    for col in parents_df.columns:
        if col not in mutants_df.columns:
            mutants_df[col] = None

    df_out = pd.concat([mutants_df, parents_df], ignore_index=True)

    # 统计
    ddg_values = mutants_df['total_ddg']
    stabilizing = (ddg_values < 0).sum()
    destabilizing = (ddg_values > 0).sum()

    print(f"\n{'='*60}")
    print(f"✅ ddG打分完成!")
    print(f"   总变体: {len(mutants_df)}")
    print(f"   稳定化 (ddG<0): {stabilizing} ({stabilizing/max(len(mutants_df),1)*100:.1f}%)")
    print(f"   去稳定化 (ddG>0): {destabilizing} ({destabilizing/max(len(mutants_df),1)*100:.1f}%)")
    if len(ddg_values) > 0:
        print(f"   ddG范围: {ddg_values.min():.3f} ~ {ddg_values.max():.3f}")
        print(f"   ddG均值: {ddg_values.mean():.3f}")

    # 排序: 先按pred_brightness降序, 再按total_ddg升序 (更稳定的在前)
    df_out = df_out.sort_values(
        by=['pred_brightness', 'total_ddg'],
        ascending=[False, True]
    ).reset_index(drop=True)

    df_out['seq_id'] = [f"ddg_{i:05d}" for i in range(len(df_out))]
    df_out.to_csv(out_path, index=False)

    # Top 20 预览
    top20 = df_out[df_out['source'] == 'bright_mut'].head(20)
    if len(top20) > 0:
        print(f"\n🏆 Top 20 (亮度最高 + 最稳定):")
        for _, row in top20.iterrows():
            print(f"   亮度={row.get('pred_brightness', 0):.4f}, "
                  f"ddG={row.get('total_ddg', 0):+.3f}, "
                  f"突变={row.get('num_mutations_from_parent', '?')}, "
                  f"类型={row.get('gfp_type', '?')}")

    print(f"\n💾 存档: {out_path}")
    print(f"💡 Phase 4 将基于此存档进行最终筛选")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: ESM3-ddG 热稳定性打分")
    parser.add_argument("--force", action="store_true", help="强制重跑")
    args = parser.parse_args()
    score_stability(force=args.force)
