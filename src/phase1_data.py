"""
Phase 1: 数据清洗与 ESM-2 特征提取 (含 PLL 热稳代理)
包含完整的 ESM-2 模型加载、Batch 推理和特征保存逻辑。
"""
import re
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# 导入全局配置
from src.config import (
    RAW_DIR, PROCESSED_DIR, FEATS_DIR, AA_TO_INT, WT_AVGFP, 
    AUTHOR, DATE_STR, seed_everything, get_device
)

# ================= 1. 数据清洗辅助函数 =================
def parse_mutations(mut_str: str, wt_seq: str) -> str:
    """
    解析突变字符串。
    采用“强制突变”策略，兼容各种非标准格式，并自动过滤终止密码子。
    """
    if pd.isna(mut_str) or str(mut_str).strip().upper() in ["WT", ""]:
        return wt_seq
    
    seq_list = list(wt_seq)
    
    # 使用正则表达式提取所有形如 "字母+数字+字母/星号" 的突变
    # 兼容 A109D, a109d, A109*, A109D:N145D 等格式
    mutations = re.findall(r'([A-Za-z])(\d+)([A-Za-z*])', str(mut_str))
    
    # 如果不是 WT 且没提取到任何有效突变，说明格式彻底坏了，丢弃
    if not mutations:
        return None 
        
    for wt_aa, pos_str, mut_aa in mutations:
        pos = int(pos_str) - 1  # 1-based 转 0-based
        mut_aa = mut_aa.upper()
        
        # 比赛规则：严禁包含终止密码子 (*)
        if mut_aa == '*':
            return None
            
        # 只要位置合法，直接强制突变 (Force Mutation)
        # 放弃 seq_list[pos] == wt_aa 的严格校验，包容数据集的参考基准差异
        if 0 <= pos < len(seq_list):
            seq_list[pos] = mut_aa 
        else:
            # 位置越界（比如序列只有238个，但突变写到了300），说明数据完全错乱，丢弃
            return None 
            
    return "".join(seq_list)

def prepare_full_data():
    """清洗 14 万条原始数据，生成 full_sequences_full.csv (保留 Exclusion 数据用于训练)"""
    print("\n" + "="*50)
    print("🧹 [Phase 1.1] 开始深度清洗全量数据")
    print("="*50)
    
    raw_path = RAW_DIR / "GFP_data.xlsx"
    if not raw_path.exists():
        raise FileNotFoundError(f"❌ 找不到原始数据文件: {raw_path}")
        
    df = pd.read_excel(raw_path)
    initial_count = len(df)
    print(f"-> 1. 读取原始数据: {initial_count} 条")
    
    # 2. 解析全长序列 (强制突变策略)
    print("-> 2. 解析突变序列 (1-based 转 0-based)...")
    df['full_sequence'] = [parse_mutations(x, WT_AVGFP) for x in tqdm(df['aaMutations'], desc="解析中")]
    df = df.dropna(subset=['full_sequence'])
    print(f"   -> 解析成功: {len(df)} 条 (丢弃了格式错误或包含终止密码子的序列)")
    
    # 3. 序列去重 (关键步骤：不同的突变字符串可能对应相同的氨基酸序列)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['full_sequence'])
    print(f"-> 3. 序列去重: {before_dedup} -> {len(df)} 条 (剔除了 {before_dedup - len(df)} 条重复氨基酸序列)")

    # 4. 死值与异常值清洗
    before_clean = len(df)
    
    # 4.1 剔除出现次数过多的“死值” (如仪器底噪、测量失败的固定值)
    brightness_counts = df['Brightness'].value_counts()
    dead_values = brightness_counts[brightness_counts > 50].index
    df = df[~df['Brightness'].isin(dead_values)]
    
    # 4.2 剔除物理上不可能的负数亮度
    df = df[df['Brightness'] >= 0]
    
    print(f"-> 4. 死值与异常值清洗: {before_clean} -> {len(df)} 条 (剔除了 {before_clean - len(df)} 条无效测量值)")
        
    # 5. 保存 (注意：这里不剔除 Exclusion List，全量用于训练！)
    out_path = PROCESSED_DIR / "full_sequences_full.csv"
    df[['full_sequence', 'Brightness']].to_csv(out_path, index=False)
    
    print("\n" + "-"*50)
    print(f"✅ 清洗完成！最终有效训练数据: {len(df)} 条")
    print(f"💡 提示: Exclusion List 数据已保留用于训练。将在最终提交阶段 (Phase 4) 进行过滤。")
    print(f"💾 保存至: {out_path}")
    print("-"*50)
    return out_path


# ================= 2. 物理化学特征计算 =================
def calculate_physicochemical(seqs: list) -> np.ndarray:
    """计算 pI, GRAVY, MW 作为抗聚集代理特征"""
    feats = []
    for seq in tqdm(seqs, desc="计算理化特征"):
        try:
            prot = ProteinAnalysis(seq)
            feats.append([prot.isoelectric_point(), prot.gravy(), prot.molecular_weight()])
        except Exception:
            feats.append([0.0, 0.0, 0.0]) 
    return np.array(feats, dtype=np.float32)


# ================= 3. ESM-2 特征提取核心逻辑 =================
def extract_full_features(force: bool = False):
    """
    使用 ESM-2 (650M) 提取 Embedding 和 PLL (热稳代理)。
    支持断点续跑：如果检测到已存在的特征文件，将直接跳过。
    """
    seed_everything()
    device = get_device()
    
    print("\n" + "="*50)
    print(f"🧬 [Phase 1.2] ESM-2 特征提取 (设备: {device})")
    print("="*50)
    
    # 断点续跑检测
    existing_feats = list(FEATS_DIR.glob(f"GFP_Feat_Full_ESM2t33_*_{AUTHOR}_*.joblib"))
    if existing_feats and not force:
        print(f"⚠️ 检测到本地已存在特征存档: {existing_feats[0].name}")
        print("💡 自动跳过特征提取。如需强制重跑，请在 main.py 中添加 --force 参数。")
        return existing_feats[0]

    # 1. 加载清洗后的数据
    csv_path = PROCESSED_DIR / "full_sequences_full.csv"
    if not csv_path.exists():
        raise FileNotFoundError("❌ 未找到清洗后的数据，请先运行 `python main.py prep`")
        
    df = pd.read_csv(csv_path)
    seqs = df['full_sequence'].tolist()
    brightness = df['Brightness'].values.astype(np.float32) 
    print(f"-> 加载了 {len(seqs)} 条序列")

    # 2. 加载 ESM-2 模型 (优先使用本地相对路径)
    from src.config import LOCAL_ESM2_DIR # 确保导入
    
    # 判断本地相对路径是否存在
    if LOCAL_ESM2_DIR.exists():
        # 将 Path 对象转换为字符串传给 transformers 库
        model_name = str(LOCAL_ESM2_DIR) 
        print(f"-> 检测到本地模型目录，正在加载: {model_name}")
    else:
        model_name = "facebook/esm2_t33_650M_UR50D" 
        print(f"-> 本地目录不存在 ({LOCAL_ESM2_DIR})，正在从 HuggingFace 下载: {model_name}")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    print("-> 模型加载完成！")


    # ================= 通道 1: ESM-2 Embedding (Mean Pooling) =================
    print("\n-> [1/3] 提取 ESM-2 Embedding (全局表征)...")
    embeddings = []
    batch_size = 16  # 4080 16G 跑 650M，batch_size 16 很稳
    
    for i in tqdm(range(0, len(seqs), batch_size), desc="Embedding 推理"):
        batch_seqs = seqs[i:i+batch_size]
        # tokenize 并 padding 到 batch 内最长 (最大 250)
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=250).to(device)
        
        with torch.no_grad():
            # 获取最后一层隐藏状态
            outputs = model.esm(**inputs) if hasattr(model, 'esm') else model(**inputs, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state  # Shape: [batch, seq_len, 1280]
            
            # Mean Pooling: 排除 padding token，对有效 token 求平均
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # Shape: [batch, seq_len, 1]
            emb = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            embeddings.append(emb.cpu().numpy())
            
    X_emb = np.vstack(embeddings).astype(np.float32)
    print(f"   -> Embedding 提取完成，Shape: {X_emb.shape}")

    # ================= 通道 2: PLL (伪对数似然，热稳代理) =================
    print("\n-> [2/3] 计算 Pseudo-Log-Likelihood (PLL, 热稳代理)...")
    scores = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="PLL 推理"):
        batch_seqs = seqs[i:i+batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=250).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits  # Shape: [batch, seq_len, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 提取真实 token 对应的 log prob
            true_log_probs = torch.gather(log_probs, 2, inputs['input_ids'].unsqueeze(-1)).squeeze(-1)
            mask = inputs['attention_mask'].bool()
            
            for j in range(len(batch_seqs)):
                valid_mask = mask[j].clone()
                valid_mask[0] = False  # 排除 <cls>
                last_idx = valid_mask.nonzero(as_tuple=True)[0][-1]
                valid_mask[last_idx] = False  # 排除 <eos>
                
                if valid_mask.sum() > 0:
                    scores.append(true_log_probs[j][valid_mask].mean().item())
                else:
                    scores.append(0.0)
                    
    X_pll = np.array(scores, dtype=np.float32).reshape(-1, 1)
    print(f"   -> PLL 计算完成，Shape: {X_pll.shape}")

    # ================= 通道 3: 物理化学特征 =================
    print("\n-> [3/3] 计算物理化学特征 (抗聚集代理)...")
    X_phys = calculate_physicochemical(seqs)
    print(f"   -> 理化特征计算完成，Shape: {X_phys.shape}")

    # ================= 融合特征与保存 =================
    print("\n-> 融合三通道特征...")
    X_combined = np.hstack([X_emb, X_pll, X_phys])
    print(f"   -> 融合后总特征 Shape: {X_combined.shape}")
    
    # 序列整数编码 (用于溯源)
    max_len = max(len(s) for s in seqs)
    seqs_int = np.zeros((len(seqs), max_len), dtype=np.int8)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            seqs_int[i, j] = AA_TO_INT.get(aa, 0)

    # 组装字典 (严格参考 diy规范.txt 的格式)
    out_dict = {
        "X": X_combined,
        "X_emb": X_emb,      
        "X_pll": X_pll,      
        "X_phys": X_phys,
        "Y": brightness,
        "seqs": seqs_int,
        "metadata": {
            "extract_date": DATE_STR,
            "model_version": model_name,
            "feature_dim_combined": X_combined.shape[1],
            "sample_size": len(seqs),
            "is_full_dataset": True
        }
    }
    
    filename = f"GFP_Feat_Full_ESM2t33_PLL_Phys_{AUTHOR}_{DATE_STR}_v1.joblib"
    out_path = FEATS_DIR / filename
    joblib.dump(out_dict, out_path)
    print(f"\n✅ 特征提取全部完成！存档已保存至: {out_path}")
    return out_path
