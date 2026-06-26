"""
Phase 1.9: ESM-2 LoRA 微调 (V9 - 72k数据 + 深层MLP头)
基于V6 LoRA经验 + V7冻结MLP策略:
  - 数据: 72k优质数据 (v7_72k_quality_data.csv)
  - LoRA: r=16, alpha=32 (减少过拟合)
  - MLP头: V7深层架构 (1024→768→512→256→128→1) + LayerNorm
  - 联合损失: delta HuberLoss + 绝对亮度 MSE
  - 30 epoch + 早停

产出: data/Models/ESM2_LoRA_V9_72k_{AUTHOR}_{DATE}/
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
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd

from src.config import (
    PROCESSED_DIR, MODELS_DIR, LOCAL_ESM2_DIR,
    RANDOM_SEED, AUTHOR, DATE_STR, seed_everything, get_device
)

# ======================== 常量 ========================
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


# 与V7式深层MLP头
class ESM2LoRARegressorV9(nn.Module):
    """V9模型: LoRA + LayerNorm + 深层MLP头 (V7架构)"""
    def __init__(self, base_model, num_gfp_types=4):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size  # 1280
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
        pooled = self.pool_norm(pooled)  # LayerNorm

        type_onehot = torch.zeros(pooled.size(0), self.num_gfp_types, device=pooled.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)

        combined = torch.cat([pooled, type_onehot], dim=1)
        return self.regressor(combined).squeeze(-1)


def fine_tune_lora_v9(force: bool = False):
    """V9: LoRA + 72k优质数据 + 深层MLP头"""
    seed_everything()
    device = get_device()
    print(f"\n🔥 [Phase 1.9] ESM-2 LoRA V9 (72k数据 + 深层MLP头) | 设备: {device}")

    out_dir = MODELS_DIR / f"ESM2_LoRA_V9_72k_{AUTHOR}_{DATE_STR}"
    if out_dir.exists() and not force:
        print(f"⚠️ 检测到已存在权重: {out_dir}。如需重跑请加 --force")
        return

    # ================= 1. 加载72k质量数据 =================
    data_path = PROCESSED_DIR / "v7_72k_quality_data.csv"
    if not data_path.exists():
        print("❌ 72k数据不存在! 请先运行: python main.py prepare_72k")
        return

    df = pd.read_csv(data_path)
    print(f"\n-> 加载72k训练数据: {len(df)} 条")

    # 与V5完全相同: 从brightness和gfp_type计算delta
    seqs = df['sequence'].tolist()
    gfp_types_list = df['gfp_type'].tolist()
    brightness = df['brightness'].values

    labels = np.array([b - WT_BRIGHTNESS[gt] for b, gt in zip(brightness, gfp_types_list)],
                      dtype=np.float32)
    type_indices = np.array([GFP_TYPE_TO_IDX[gt] for gt in gfp_types_list],
                            dtype=np.int64)

    print(f"  亮度范围: {brightness.min():.3f} ~ {brightness.max():.3f}")
    print(f"  Delta范围: {labels.min():.3f} ~ {labels.max():.3f}")
    for gt in GFP_TYPE_LIST:
        mask = [t == gt for t in gfp_types_list]
        deltas = labels[mask]
        if len(deltas) > 0:
            print(f"  {gt}: N={len(deltas)}, delta={deltas.mean():+.3f}±{deltas.std():.3f}")

    # 与V7一致: 80/10/10划分
    X_tv, X_test, Y_tv, Y_test, T_tv, T_test = train_test_split(
        seqs, labels, type_indices, test_size=0.1, random_state=RANDOM_SEED
    )
    X_train, X_val, Y_train, Y_val, T_train, T_val = train_test_split(
        X_tv, Y_tv, T_tv, test_size=0.111, random_state=RANDOM_SEED
    )

    # ================= 3. 过采样 (与V5完全一致) =================
    print("\n-> 正在对极端亮度样本进行过采样 (同V5策略)...")
    X_train_list = list(X_train)
    Y_train_list = list(Y_train)
    T_train_list = list(T_train)

    # delta > 0.5 过采样2x
    high_brightness_indices = [i for i, y in enumerate(Y_train_list) if y > 0.5]
    for idx in high_brightness_indices:
        X_train_list.append(X_train_list[idx])
        Y_train_list.append(Y_train_list[idx])
        T_train_list.append(T_train_list[idx])
    print(f"   delta>0.5: {len(high_brightness_indices)} 条 × 2")

    # delta < -0.5 过采样1.5x
    low_brightness_indices = [i for i, y in enumerate(Y_train_list) if y < -0.5]
    for idx in low_brightness_indices[:len(low_brightness_indices)//2]:
        X_train_list.append(X_train_list[idx])
        Y_train_list.append(Y_train_list[idx])
        T_train_list.append(T_train_list[idx])
    print(f"   delta<-0.5: {len(low_brightness_indices)//2} 条 × 1.5")

    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    T_train = np.array(T_train_list)
    print(f"   过采样后: {len(X_train)} 条")

    # ================= 4. 模型加载 =================
    model_name = str(LOCAL_ESM2_DIR) if LOCAL_ESM2_DIR.exists() else "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16, lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05, bias="none"
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    model = ESM2LoRARegressorV9(peft_model, num_gfp_types=len(GFP_TYPE_LIST)).to(device)
    # 使用非重入式gradient checkpointing (兼容LoRA + PyTorch 2.x)
    model.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ================= 5. DataLoader =================
    batch_size = 48  # LoRA+深层头需要更多显存
    train_loader = DataLoader(
        GFPDatasetWithType(X_train, Y_train, T_train, tokenizer),
        batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        GFPDatasetWithType(X_val, Y_val, T_val, tokenizer),
        batch_size=batch_size * 2, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        GFPDatasetWithType(X_test, Y_test, T_test, tokenizer),
        batch_size=batch_size * 2, shuffle=False, num_workers=4
    )

    # ================= 6. 优化器 & 损失 =================
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5, weight_decay=0.01
    )
    epochs = 30
    num_training_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )

    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.HuberLoss(delta=0.5, reduction='none')

    # ================= 7. 训练循环 (早停) =================
    print(f"\n{'='*60}")
    print(f"开始训练 V9 (72k数据 + LoRA r=16 + 深层MLP头 | {epochs} epochs)")
    print(f"{'='*60}")
    print(f"  训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
    print(f"  加权: delta>0.5→6x, 中间→2x, delta<-0.5→2.5x")

    best_val_r2 = -999
    best_epoch = -1
    no_improve = 0
    patience = 8
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            gfp_type = batch["gfp_type"].to(device)

            with torch.amp.autocast('cuda'):
                preds = model(input_ids, attention_mask, gfp_type)

                # 与V5完全一致的加权策略
                weights = torch.ones_like(labels)
                weights = torch.where(labels > 0.5, torch.tensor(6.0, device=device), weights)
                weights = torch.where((labels >= -0.5) & (labels <= 0.5), torch.tensor(2.0, device=device), weights)
                weights = torch.where(labels < -0.5, torch.tensor(2.5, device=device), weights)

                loss = (criterion(preds, labels) * weights).mean()

            scaler.scale(loss).backward()
            
            # 调试: 检查LoRA参数是否有梯度
            if epoch == 0 and batch_idx == 0:
                lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
                if lora_params:
                    grad_norms = [p.grad.norm().item() for p in lora_params if p.grad is not None]
                    none_count = sum(1 for p in lora_params if p.grad is None)
                    print(f"\n  [DEBUG] LoRA params: {len(lora_params)}, with_grad: {len(grad_norms)}, None: {none_count}")
                    if grad_norms:
                        print(f"    Grad norms range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
                    else:
                        print("    ⚠️ WARNING: All LoRA gradients are None!")
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # 验证
        model.eval()
        val_preds_all, val_labels_all, val_types_all = [], [], []

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
                val_types_all.extend(gfp_type.cpu().numpy())

        val_preds = np.array(val_preds_all)
        val_labels = np.array(val_labels_all)
        val_types = np.array(val_types_all)
        val_r2 = r2_score(val_labels, val_preds)
        val_ccc = concordance_correlation_coefficient(val_labels, val_preds)
        val_mae = mean_absolute_error(val_labels, val_preds)
        val_r, _ = pearsonr(val_labels, val_preds)

        # 高亮度区间: delta+WT >= 4.0
        hi_mask = np.array([
            labels_i + WT_BRIGHTNESS[GFP_TYPE_LIST[int(t_i)]] >= 4.0
            for labels_i, t_i in zip(val_labels, val_types)
        ])
        if hi_mask.sum() > 5:
            hi_r2 = r2_score(val_labels[hi_mask], val_preds[hi_mask])
            hi_mae = mean_absolute_error(val_labels[hi_mask], val_preds[hi_mask])
            hi_r, _ = pearsonr(val_labels[hi_mask], val_preds[hi_mask])
        else:
            hi_r2, hi_mae, hi_r = float('nan'), float('nan'), float('nan')

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | "
                  f"R²: {val_r2:.4f} | r: {val_r:.4f} | CCC: {val_ccc:.4f} | MAE: {val_mae:.4f}")
            print(f"    高亮度(brightness>=4.0, N={hi_mask.sum()}): "
                  f"R²={hi_r2:.4f}, r={hi_r:.4f}, MAE={hi_mae:.4f}")
        else:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | CCC: {val_ccc:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            no_improve = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            regressor_state = {k.replace('regressor.', ''): v
                               for k, v in best_state.items()
                               if k.startswith('regressor.')}
            torch.save(regressor_state, out_dir / "regressor.pt")
            print(f"    ✔ 新最佳 (R²={val_r2:.4f}, r={val_r:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  ⏸ 早停: {patience} epoch无改善 (最佳R²={best_val_r2:.4f}, Epoch {best_epoch})")
            break

    print(f"\n✅ V9训练完成! 最佳 Epoch {best_epoch}, Val R²: {best_val_r2:.4f}")

    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    test_preds_all, test_labels_all, test_types_all = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            gfp_type = batch["gfp_type"].to(device)

            with torch.amp.autocast('cuda'):
                preds = model(input_ids, attention_mask, gfp_type)
            test_preds_all.extend(preds.cpu().numpy())
            test_labels_all.extend(labels.cpu().numpy())
            test_types_all.extend(gfp_type.cpu().numpy())

    test_preds = np.array(test_preds_all)
    test_labels = np.array(test_labels_all)
    test_types = np.array(test_types_all)
    test_r2 = r2_score(test_labels, test_preds)
    test_mae = mean_absolute_error(test_labels, test_preds)
    test_r, _ = pearsonr(test_labels, test_preds)
    print(f"\n📊 测试集: R²={test_r2:.4f}, Pearson r={test_r:.4f}, MAE={test_mae:.4f}")

    # 绝对亮度评估
    test_abs_pred = np.array([d + WT_BRIGHTNESS[GFP_TYPE_LIST[int(t)]] for d, t in zip(test_preds, test_types)])
    test_abs_true = np.array([d + WT_BRIGHTNESS[GFP_TYPE_LIST[int(t)]] for d, t in zip(test_labels, test_types)])
    print(f"📊 测试集 (绝对亮度): R²={r2_score(test_abs_true, test_abs_pred):.4f}, "
          f"r={pearsonr(test_abs_true, test_abs_pred)[0]:.4f}, "
          f"MAE={mean_absolute_error(test_abs_true, test_abs_pred):.4f}")

    # 分类型测试
    for gt_idx, gt in enumerate(GFP_TYPE_LIST):
        mask = test_types == gt_idx
        if mask.sum() < 5:
            continue
        gt_preds = test_preds[mask]
        gt_labels = test_labels[mask]
        gt_r2 = r2_score(gt_labels, gt_preds)
        gt_mae = mean_absolute_error(gt_labels, gt_preds)
        gt_r, _ = pearsonr(gt_labels, gt_preds)
        gt_abs_pred = gt_preds + WT_BRIGHTNESS[gt]
        gt_abs_true = gt_labels + WT_BRIGHTNESS[gt]
        gt_r2_abs = r2_score(gt_abs_true, gt_abs_pred)
        print(f"  {gt}: R²_delta={gt_r2:.4f}, R²_abs={gt_r2_abs:.4f}, r={gt_r:.4f}, MAE={gt_mae:.4f} (N={mask.sum()})")

    # 高亮度区间
    hi_mask = np.array([
        test_labels[i] + WT_BRIGHTNESS[GFP_TYPE_LIST[int(test_types[i])]] >= 4.0
        for i in range(len(test_labels))
    ])
    if hi_mask.sum() > 10:
        hi_r2 = r2_score(test_labels[hi_mask], test_preds[hi_mask])
        hi_r, _ = pearsonr(test_labels[hi_mask], test_preds[hi_mask])
        hi_r2_abs = r2_score(test_abs_true[hi_mask], test_abs_pred[hi_mask])
        print(f"  高亮度(>=4.0): R²_delta={hi_r2:.4f}, R²_abs={hi_r2_abs:.4f}, r={hi_r:.4f} (N={hi_mask.sum()})")

    # ================= 9. 保存配置 =================
    joblib.dump({
        'wt_brightness': WT_BRIGHTNESS,
        'gfp_type_list': GFP_TYPE_LIST,
        'gfp_type_to_idx': GFP_TYPE_TO_IDX,
        'version': 'V9_72k_LoRA_DeepMLP',
    }, out_dir / "model_config.joblib")

    print(f"\n💾 V9模型已保存至: {out_dir}")
    print(f"💡 预测: 最终亮度 = 模型预测(delta) + WT亮度")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    fine_tune_lora_v9(force=force)
