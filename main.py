"""
项目统一命令行入口 (支持 Phase 1 全流程 + LoRA 微调)
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

def run_prep(args):
    from src.phase1_data import prepare_full_data
    prepare_full_data()

def run_extract(args):
    from src.phase1_data import extract_full_features
    extract_full_features(force=args.force)

def run_train(args):
    from src.phase1_train import train_models
    train_models(force=args.force)

def run_finetune(args):
    from src.phase1_lora_finetune import finetune_lora
    finetune_lora(force=args.force)

def run_generate(args):
    from src.phase2_generate import generate_sequences
    generate_sequences(num_samples=args.num, mask_ratio=args.mask, force=args.force)

def run_score_ddg(args):
    from src.phase3_ddg_scoring import score_stability
    score_stability(force=args.force)

def run_score(args):
    from src.phase4_scoring import score_and_filter
    score_and_filter(force=args.force)

def run_prepare_v6(args):
    from src.prepare_v6_data import prepare_v6_data
    prepare_v6_data(force=args.force)

def run_train_v6(args):
    from src.phase1_lora_finetune_v6 import fine_tune_lora_v9
    fine_tune_lora_v9(force=args.force)

def run_train_frozen(args):
    from src.phase1_frozen_esm2_mlp import main as frozen_main
    frozen_main()

def run_prepare_72k(args):
    from src.prepare_72k_data import prepare_72k_data
    prepare_72k_data(force=args.force)

def main():
    parser = argparse.ArgumentParser(description="GFP 六边形战士流水线")
    subparsers = parser.add_subparsers(dest="task", help="可用任务")
    
    parser_prep = subparsers.add_parser("prep", help="Phase 1.1: 清洗原始数据")
    parser_prep.add_argument("--force", action="store_true")
    
    parser_extract = subparsers.add_parser("extract", help="Phase 1.2: ESM2 特征提取 (需 GPU)")
    parser_extract.add_argument("--force", action="store_true")
    
    parser_train = subparsers.add_parser("train", help="Phase 1.3: 传统模型训练 (XGB/MLP/Stacking)")
    parser_train.add_argument("--force", action="store_true")

    # 🌟 新增 LoRA 微调命令
    parser_finetune = subparsers.add_parser("finetune", help="Phase 1.4: ESM2 LoRA 端到端微调 (需 GPU, 耗时较长)")
    parser_finetune.add_argument("--force", action="store_true")

    parser_gen = subparsers.add_parser("generate", help="Phase 2: 高亮度序列+随机突变+LoRA筛选 (需 GPU)")
    parser_gen.add_argument("--force", action="store_true")
    parser_gen.add_argument("--num", type=int, default=5000, help="生成总量 (top_n*variants_per_seq)")
    parser_gen.add_argument("--mask", type=float, default=0.15, help="(兼容保留)")

    # Phase 3: ESM3-ddG 热稳定性打分
    parser_ddg = subparsers.add_parser("score_ddg", help="Phase 3: ESM3-ddG 热稳定性打分 (需 GPU, ~5.6GB模型)")
    parser_ddg.add_argument("--force", action="store_true")

    parser_score = subparsers.add_parser("score", help="Phase 4: 综合筛选(brightness+ddG) + ESM-2表面优化 → Top 20")
    parser_score.add_argument("--force", action="store_true")

    # V6: 高亮度专家模型
    parser_pv6 = subparsers.add_parser("prepare_v6", help="V6数据: 高亮度聚焦采样 → v6_training_data.csv")
    parser_pv6.add_argument("--force", action="store_true")

    parser_tv6 = subparsers.add_parser("train_v6", help="V9训练: LoRA+72k数据+深层MLP头 (需GPU)")
    parser_tv6.add_argument("--force", action="store_true")

    # 72k优质数据
    parser_72k = subparsers.add_parser("prepare_72k", help="72k优质数据筛选 (从141k中分层采样)")
    parser_72k.add_argument("--force", action="store_true")

    # Frozen ESM-2 + MLP (不用LoRA, 避开梯度检查点)
    parser_frozen = subparsers.add_parser("train_frozen", help="冻结ESM-2特征 + MLP训练 (无需LoRA)")
    parser_frozen.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.task == "prep": run_prep(args)
    elif args.task == "extract": run_extract(args)
    elif args.task == "train": run_train(args)
    elif args.task == "finetune": run_finetune(args)
    elif args.task == "generate": run_generate(args)
    elif args.task == "score_ddg": run_score_ddg(args)
    elif args.task == "score": run_score(args)
    elif args.task == "prepare_v6": run_prepare_v6(args)
    elif args.task == "train_v6": run_train_v6(args)
    elif args.task == "prepare_72k": run_prepare_72k(args)
    elif args.task == "train_frozen": run_train_frozen(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
