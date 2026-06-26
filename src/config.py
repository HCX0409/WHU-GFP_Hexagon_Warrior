"""
全局配置模块
解决路径硬编码、随机种子不统一、跨平台兼容问题。
"""
import os
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ================= 1. 路径配置 (杜绝硬编码) =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATS_DIR = DATA_DIR / "Feats"
MODELS_DIR = DATA_DIR / "Models"  # 存放训练好的 XGBoost 等小模型
GEN_SEQS_DIR = PROJECT_ROOT / "generated_seqs"

# 【新增】预训练大模型存放目录 (如 ESM-2)
PRETRAINED_MODELS_DIR = PROJECT_ROOT / "models" 

# 自动创建目录
for d in [RAW_DIR, PROCESSED_DIR, FEATS_DIR, MODELS_DIR, GEN_SEQS_DIR, PRETRAINED_MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 【新增】ESM-2 本地模型路径配置 (支持环境变量与相对路径 fallback)
# 优先读取环境变量 LOCAL_ESM2_PATH，如果没有设置，则使用项目根目录下的 models/esm2_t33_650M_UR50D
_env_esm2_path = os.environ.get("LOCAL_ESM2_PATH")
if _env_esm2_path:
    LOCAL_ESM2_DIR = Path(_env_esm2_path)
else:
    LOCAL_ESM2_DIR = PRETRAINED_MODELS_DIR / "esm2_t33_650M_UR50D"

# ESM-3 ddG 模型路径 (热稳定性预测, Phase 3)
_env_esm3_ddg_path = os.environ.get("LOCAL_ESM3_DDG_PATH")
if _env_esm3_ddg_path:
    LOCAL_ESM3_DDG_DIR = Path(_env_esm3_ddg_path)
else:
    LOCAL_ESM3_DDG_DIR = PRETRAINED_MODELS_DIR / "esm3_ddg_v2"

# ================= 2. 全局常量 =================
RANDOM_SEED = 42
AUTHOR = "HCX"  # 请修改为您自己的名字缩写
DATE_STR = datetime.now().strftime("%y%m%d")

# 氨基酸整数编码
AA_TO_INT = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

# avGFP 野生型序列 (238 aa, 训练数据基准)
WT_AVGFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

# sfGFP 野生型序列 (238 aa, 本赛季比赛基准, PDB: 2B3P)
WT_SFGFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNIVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

# ================= 3. 工具函数 =================
def seed_everything(seed: int = RANDOM_SEED):
    """固定所有随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """自动检测最佳设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
