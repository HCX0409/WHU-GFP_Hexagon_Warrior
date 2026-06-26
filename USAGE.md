# 项目复现指南 / Reproduction Guide

本文档说明如何在**不同硬件与系统环境**下复现本项目全部功能。

---

## 一、硬件要求

| 配置等级 | GPU显存 | 内存(RAM) | 磁盘 | 说明 |
|:---:|:---:|:---:|:---:|------|
| **推荐** | ≥16GB (如 RTX 4080) | ≥32GB | ≥20GB | 可完整运行全部训练与推理 |
| **最低** | ≥8GB (如 RTX 3060) | ≥16GB | ≥15GB | 需调小 `batch_size`，训练耗时翻倍 |
| **仅CPU** | 无GPU | ≥16GB | ≥15GB | 可推理，**训练极慢**（仅建议测试流程） |

> **磁盘说明**: ESM-2模型(~2.6GB) + ESM3-ddG模型(~5.6GB) + 原始数据(~37MB) + 生成数据(~数百MB)

---

## 二、软件环境

### 2.1 系统支持
- **Windows 10/11** (本项目开发环境)
- **Linux** (Ubuntu 20.04+)
- **macOS** (Apple Silicon可通过MPS加速)

代码已内置设备自动检测：CUDA → MPS (Mac) → CPU。

### 2.2 创建Conda环境

```bash
# 1. 创建环境 (Python 3.10)
conda create -n gfp_Hexagon_Warrior python=3.10 -y
conda activate gfp_Hexagon_Warrior

# 2. 安装PyTorch (根据CUDA版本选择)
# NVIDIA GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# NVIDIA GPU (CUDA 12.x):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 仅CPU:
pip install torch torchvision torchaudio
# Mac (MPS加速):
pip install torch torchvision torchaudio

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 2.3 Windows特殊处理

如遇到 `KMP_DUPLICATE_LIB_OK` 报错：
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
```
或在Python代码开头添加：
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

---

## 三、模型下载

### 3.1 ESM-2 (650M参数，~2.6GB)

```bash
# 方法一：HuggingFace官方 (需科学上网)
# 代码会自动下载到 models/esm2_t33_650M_UR50D/

# 方法二：使用国内镜像 (推荐国内用户)
set HF_ENDPOINT=https://hf-mirror.com
```

或手动下载到 `models/esm2_t33_650M_UR50D/` 目录：
- 从 https://huggingface.co/facebook/esm2_t33_650M_UR50D 下载全部文件

### 3.2 ESM3-ddG v2 (~1.4B参数，~5.6GB)

```bash
# 同上，设置镜像或直接下载
# 目标目录：models/esm3_ddg_v2/
```
- 从 https://huggingface.co/hazemessam/esm3_ddg_v2 下载全部文件

### 3.3 环境变量 (可选)

如果模型放在其他位置，可通过环境变量指定路径：
```bash
set LOCAL_ESM2_PATH=D:\path\to\esm2_t33_650M_UR50D
set LOCAL_ESM3_DDG_PATH=D:\path\to\esm3_ddg_v2
```

---

## 四、硬件参数调优

**核心原则**: GPU显存不足时，减小 `batch_size`；内存不足时，减小 `num_workers`。

### 4.1 各训练脚本可调参数

| 脚本 | 参数 | 默认值(16GB GPU) | 8GB GPU建议 | 仅CPU建议 |
|------|------|:---:|:---:|:---:|
| `phase1_lora_finetune_v5.py` (V5训练) | `batch_size` | 72 | 32~40 | 8~16 |
| | `num_workers` | 4 | 2~4 | 0 |
| `phase1_lora_finetune_v6.py` (V9训练) | `batch_size` | 48 | 24~32 | 8~16 |
| | `num_workers` | 8 | 2~4 | 0 |
| `phase1_frozen_esm2_mlp.py` (V7训练) | `batch_size` (嵌入提取) | 48~64 | 24~32 | 16 |
| | `batch_size` (MLP训练) | 256 | 128~256 | 128 |
| | `OMP_NUM_THREADS` | 8 | 4 | CPU核数 |

### 4.2 如何修改

打开对应训练脚本，找到参数行直接修改：

```python
# phase1_lora_finetune_v6.py 第210行
batch_size = 48  # ← 改为适合您GPU显存的值

# 第213行
num_workers=8  # ← 内存不足时减小此值，CPU训练设为0
```

### 4.3 显存不足 (OOM) 时的处理

1. **首选**: 减小 `batch_size`（减半即可）
2. **确认** gradient checkpointing 已启用（V5/V9默认已开启）
3. **减小** `num_workers`（释放内存）
4. **设置环境变量**：`set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

---

## 五、执行流程

### 5.1 完整流程 (从零开始)

```bash
# 激活环境
conda activate gfp_Hexagon_Warrior

# ---- Phase 1: 数据准备与模型训练 ----
python main.py prep                          # 1.1 清洗原始数据 (<1min)
python main.py finetune --force              # 1.2 V5训练: 全量122k LoRA (~6h@4080)
python main.py prepare_72k --force           # 1.3 72k优质数据采样 (~5min)
python main.py train_v6 --force              # 1.4 V9训练: 72k LoRA+深层MLP (~4h@4080)
python main.py train_frozen --force          # 1.5 V7训练: 冻结ESM-2+MLP (~1h@4080)

# ---- Phase 2: 序列生成与筛选 ----
python main.py generate --force              # 分层突变+共识筛选5轮 (~10min)

# ---- Phase 3: 热稳定性验证 ----
python main.py score_ddg --force             # ESM3-ddG打分 (~6h@4080)

# ---- Phase 4: 综合筛选 ----
python main.py score --force                 # 最终Top 20 (<1min)
```

### 5.2 仅推理 (使用已训练模型)

```bash
python scripts/predict_v9.py                  # V9单序列亮度预测
python scripts/test_model.py                  # 交互式多模型预测
python scripts/scatter_v9.py                  # 生成V9散点图
python scripts/scatter_v7.py                  # 生成V7散点图
python scripts/final_pipeline.py              # 四数据源合并+选5条候选
python scripts/generate_doc.py                # 生成技术文档(docx)
```

### 5.3 各阶段耗时参考

| 阶段 | RTX 4080 (16GB) | RTX 3060 (8GB) | 仅CPU |
|------|:---:|:---:|:---:|
| V5训练 (122k) | ~6h | ~12h | ~72h |
| V9训练 (72k) | ~4h | ~8h | ~48h |
| V7训练 (MLP) | ~1h | ~2h | ~6h |
| Phase 2生成 | ~10min | ~20min | ~2h |
| Phase 3 ddG | ~6h | ~12h | ~36h |

---

## 六、预期输出

### 训练产出
```
data/Models/
├── ESM2_LoRA_V5_TypeEmbed_HCX_*/     # V5 LoRA适配器
├── ESM2_LoRA_V9_72k_HCX_*/           # V9 LoRA适配器
└── ESM2_Frozen_MLP_V7_72k_HCX_*/     # V7 MLP权重
```

### 筛选产出
```
generated_seqs/
├── all_scored_merged_HCX_*.csv        # 39,020条全量打分
├── final_5_candidates_v3_HCX_*.csv    # 最终5条候选序列
└── phase2_hierarchical_HCX_*.csv      # 分层突变中间结果
```

### 文档产出
```
GFP_Hexagon_Warrior_技术文档_v*.docx   # 竞赛技术文档
```

---

## 七、常见问题

### Q1: 没有GPU能运行吗？
可以。代码自动检测CPU运行，但训练耗时极长。建议仅用于**推理测试**：
```bash
python scripts/predict_v9.py    # 单序列预测，CPU约30秒/条
```

### Q2: 训练时显存溢出 (CUDA Out of Memory)？
修改对应训练脚本的 `batch_size`，减半即可。详见第四节。

### Q3: ESM模型下载失败？
设置HuggingFace镜像：
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### Q4: Windows下 `num_workers` 报错？
Windows多进程限制，将 `num_workers` 设为 0 或 2：
```python
num_workers=0  # Windows兼容
```

### Q5: `ModuleNotFoundError: No module named 'esm'`？
ESM3-ddG依赖单独的 `esm` 包：
```bash
pip install esm
```

---

## 八、项目联系方式

- 团队: **GFP Hexagon Warrior** — WHU-CHINA
- 竞赛: SynBio Challenges 2026
- 作者: HCX
