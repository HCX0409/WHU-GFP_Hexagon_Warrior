"""
Phase 1.5: ESM-2 LoRA 微调 (V5 - 相对亮度 + GFP类型嵌入)
核心改进：
  1. 预测相对WT的亮度变化（delta），消除GFP类型系统偏差
  2. GFP类型one-hot嵌入，让模型区分不同GFP的亮度响应模式
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import pandas as pd

from src.config import (
    PROCESSED_DIR, MODELS_DIR, LOCAL_ESM2_DIR, 
    RANDOM_SEED, AUTHOR, DATE_STR, seed_everything, get_device, WT_AVGFP
)

# GFP类型 → 索引映射
GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {t: i for i, t in enumerate(GFP_TYPE_LIST)}

WT_BRIGHTNESS = {
    'avGFP': 3.7192,
    'amacGFP': 3.9707,
    'cgreGFP': 4.4969,
    'ppluGFP': 4.2258
}

def concordance_correlation_coefficient(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / (denominator + 1e-8)


class GFPDatasetWithType(Dataset):
    """带GFP类型索引的数据集"""
    def __init__(self, seqs, labels, type_indices, tokenizer, max_len=250):
        self.seqs = seqs
        self.labels = labels
        self.type_indices = type_indices
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.seqs[idx], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "gfp_type": torch.tensor(self.type_indices[idx], dtype=torch.long)
        }


class ESM2LoRARegressorV5(nn.Module):
    """V5模型：mean_pooling + GFP类型one-hot → 回归头"""
    def __init__(self, base_model, num_gfp_types=4):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size  # 1280
        input_dim = hidden_size + num_gfp_types      # 1284
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 768), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.num_gfp_types = num_gfp_types
    
    def forward(self, input_ids, attention_mask, gfp_type_idx):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        
        # GFP类型one-hot编码
        type_onehot = torch.zeros(pooled.size(0), self.num_gfp_types, device=pooled.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        
        # 拼接特征
        combined = torch.cat([pooled, type_onehot], dim=1)
        return self.regressor(combined).squeeze(-1)


def fine_tune_lora_v5(force: bool = False):
    """V5版本：相对亮度 + GFP类型嵌入"""
    seed_everything()
    device = get_device()
    print(f"\n🔥 [Phase 1.5] ESM-2 LoRA 微调 V5 (相对亮度+类型嵌入) | 设备: {device}")
    
    out_dir = MODELS_DIR / f"ESM2_LoRA_V5_TypeEmbed_{AUTHOR}_{DATE_STR}"
    if out_dir.exists() and not force:
        print(f"⚠️ 检测到已存在权重: {out_dir}。如需重跑请加 --force")
        return

    # ================= 数据加载 =================
    raw_path = Path("data/raw/GFP_data.xlsx")
    raw_df = pd.read_excel(raw_path)
    gfp_col = raw_df.columns[1]
    
    print(f"\n-> 正在解析序列并计算相对亮度...")
    
    from src.phase1_data import parse_mutations
    
    print(f"  开始解析 {len(raw_df)} 条原始数据...")
    
    all_seqs = []
    all_deltas = []
    all_gfp_types = []
    
    for idx, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="解析序列"):
        gfp_type = row[gfp_col]
        mut_str = row['aaMutations']
        brightness = row['Brightness']
        
        seq = parse_mutations(mut_str, WT_AVGFP)
        if seq is None:
            continue
        
        wt_bright = WT_BRIGHTNESS.get(gfp_type, 3.7192)
        delta = brightness - wt_bright
        
        all_seqs.append(seq)
        all_deltas.append(delta)
        all_gfp_types.append(gfp_type)
    
    print(f"  解析完成！有效序列: {len(all_seqs)} 条")
    
    # 去重（同一序列+同一GFP类型才去重）
    unique_data = {}
    for seq, delta, gfp_type in zip(all_seqs, all_deltas, all_gfp_types):
        key = (seq, gfp_type)
        if key not in unique_data:
            unique_data[key] = {'delta': delta, 'gfp_type': gfp_type}
    
    print(f"  去重后: {len(unique_data)} 条")
    
    seqs = [k[0] for k in unique_data.keys()]
    labels = np.array([v['delta'] for v in unique_data.values()], dtype=np.float32)
    gfp_types_list = [v['gfp_type'] for v in unique_data.values()]
    type_indices = np.array([GFP_TYPE_TO_IDX.get(gt, 0) for gt in gfp_types_list], dtype=np.int64)
    
    # 统计
    print(f"\n  相对亮度统计:")
    for gfp_type in GFP_TYPE_LIST:
        mask = [gt == gfp_type for gt in gfp_types_list]
        deltas = [d for d, m in zip(labels, mask) if m]
        if deltas:
            print(f"    {gfp_type}: mean={np.mean(deltas):.3f}, std={np.std(deltas):.3f}, count={len(deltas)}")
    
    # ================= 数据集划分 =================
    X_tv, X_test, Y_tv, Y_test, T_tv, T_test = train_test_split(
        seqs, labels, type_indices, test_size=0.1, random_state=RANDOM_SEED
    )
    X_train, X_val, Y_train, Y_val, T_train, T_val = train_test_split(
        X_tv, Y_tv, T_tv, test_size=0.111, random_state=RANDOM_SEED
    )
    
    # 过采样
    print("\n-> 正在对极端亮度样本进行过采样...")
    X_train_list = list(X_train)
    Y_train_list = list(Y_train)
    T_train_list = list(T_train)
    
    high_brightness_indices = [i for i, y in enumerate(Y_train_list) if y > 0.5]
    for idx in high_brightness_indices:
        X_train_list.append(X_train_list[idx])
        Y_train_list.append(Y_train_list[idx])
        T_train_list.append(T_train_list[idx])
    
    low_brightness_indices = [i for i, y in enumerate(Y_train_list) if y < -0.5]
    for idx in low_brightness_indices[:len(low_brightness_indices)//2]:
        X_train_list.append(X_train_list[idx])
        Y_train_list.append(Y_train_list[idx])
        T_train_list.append(T_train_list[idx])
    
    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    T_train = np.array(T_train_list)
    print(f"   过采样后: {len(X_train)} 条")

    # ================= 模型加载 =================
    model_name = str(LOCAL_ESM2_DIR) if LOCAL_ESM2_DIR.exists() else "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        r=32, lora_alpha=64,
        target_modules=["query", "value", "key"],
        lora_dropout=0.1, bias="none"
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    
    model = ESM2LoRARegressorV5(peft_model, num_gfp_types=len(GFP_TYPE_LIST)).to(device)
    model.base_model.gradient_checkpointing_enable()
    
    batch_size = 72
    train_loader = DataLoader(
        GFPDatasetWithType(X_train, Y_train, T_train, tokenizer),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        GFPDatasetWithType(X_val, Y_val, T_val, tokenizer),
        batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        GFPDatasetWithType(X_test, Y_test, T_test, tokenizer),
        batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-5, weight_decay=0.01)
    num_training_steps = 12 * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )
    
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.HuberLoss(delta=0.5, reduction='none')
    epochs = 12
    
    print(f"\n{'='*60}")
    print(f"开始训练 V5 (相对亮度 + GFP类型嵌入)")
    print(f"{'='*60}")
    
    best_val_r2 = -999
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            gfp_type = batch["gfp_type"].to(device)
            
            with torch.amp.autocast('cuda'):
                preds = model(input_ids, attention_mask, gfp_type)
                
                weights = torch.ones_like(labels)
                weights = torch.where(labels > 0.5, torch.tensor(6.0, device=device), weights)
                weights = torch.where((labels >= -0.5) & (labels <= 0.5), torch.tensor(2.0, device=device), weights)
                weights = torch.where(labels < -0.5, torch.tensor(2.5, device=device), weights)
                
                loss = (criterion(preds, labels) * weights).mean()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # 验证
        model.eval()
        val_preds_all, val_labels_all = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                gfp_type = batch["gfp_type"].to(device)
                
                with torch.amp.autocast('cuda'):
                    preds = model(input_ids, attention_mask, gfp_type)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())
        
        val_preds = np.array(val_preds_all)
        val_labels = np.array(val_labels_all)
        val_r2 = r2_score(val_labels, val_preds)
        val_ccc = concordance_correlation_coefficient(val_labels, val_preds)
        val_mae = mean_absolute_error(val_labels, val_preds)
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val R²: {val_r2:.4f} | Val CCC: {val_ccc:.4f} | Val MAE: {val_mae:.4f}")
        else:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val CCC: {val_ccc:.4f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            # 立即保存
            out_dir.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            regressor_state = {k.replace('regressor.', ''): v for k, v in best_state.items() if k.startswith('regressor.')}
            torch.save(regressor_state, out_dir / "regressor.pt")
            print(f"    ✔ 新最佳模型已保存 (R²={val_r2:.4f})")
    
    print(f"\n✅ 训练完成！最佳 Val R²: {best_val_r2:.4f}")
    
    # 保存配置
    joblib.dump({
        'wt_brightness': WT_BRIGHTNESS,
        'gfp_type_list': GFP_TYPE_LIST,
        'gfp_type_to_idx': GFP_TYPE_TO_IDX
    }, out_dir / "model_config.joblib")
    
    print(f"💾 模型已保存至: {out_dir}")
    print(f"\n💡 预测时：最终亮度 = 模型预测(delta, gfp_type) + WT亮度")

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    fine_tune_lora_v5(force=force)
