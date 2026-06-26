"""
单序列亮度预测测试脚本 (物理数值对齐版)
自动校准系统误差，输出符合数据集真实物理意义的 Log10 亮度。
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from pathlib import Path

from src.config import MODELS_DIR, LOCAL_ESM2_DIR, get_device
from sklearn.linear_model import LinearRegression
import numpy as np

class ESM2LoRARegressor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # 🌟 和训练时保持一致的结构（768→512→256→1）
        self.regressor = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 768), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden_states * mask).sum(1) / mask.sum(1)
        return self.regressor(pooled).squeeze(-1)

def main():
    device = get_device()
    print(f"⚙️ 使用设备: {device}")
    
    # 1. 加载模型
    lora_dirs = list(MODELS_DIR.glob("ESM2_LoRA_*"))
    if not lora_dirs:
        print("❌ 未找到 LoRA 存档。"); return
    latest_lora_dir = max(lora_dirs, key=lambda p: p.stat().st_mtime)
    print(f"🌟 加载存档: {latest_lora_dir.name}")

    model_name = str(LOCAL_ESM2_DIR) if LOCAL_ESM2_DIR.exists() else "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(base_model, latest_lora_dir)
    
    model = ESM2LoRARegressor(peft_model).to(device)
    regressor_path = latest_lora_dir / "regressor.pt"
    if regressor_path.exists():
        model.regressor.load_state_dict(torch.load(regressor_path, map_location=device))
    model.eval()
    
    # ================= 🌟 核心修复：使用线性回归校准 =================
    print("\n-> 正在加载校准数据...")
    
    # 加载一些已知亮度的序列用于校准
    import pandas as pd
    from pathlib import Path as PathLib
    calibration_file = PathLib("data/processed/full_sequences_full.csv")
    if calibration_file.exists():
        df_cal = pd.read_csv(calibration_file)
        # 随机抽样100条用于校准
        df_sample = df_cal.sample(n=100, random_state=42)
        cal_seqs = df_sample['full_sequence'].tolist()
        cal_true = df_sample['Brightness'].values
        
        # 批量预测
        print("   -> 正在对校准集进行预测...")
        cal_preds = []
        batch_size = 16
        for i in range(0, len(cal_seqs), batch_size):
            batch_seqs = cal_seqs[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=250).to(device)
            with torch.no_grad():
                preds = model(inputs["input_ids"], inputs["attention_mask"])
            cal_preds.extend(preds.cpu().numpy())
        cal_preds = np.array(cal_preds)
        
        # 线性回归校准
        lr = LinearRegression()
        lr.fit(cal_preds.reshape(-1, 1), cal_true)
        print(f"   -> 校准完成！slope={lr.coef_[0]:.4f}, intercept={lr.intercept_:.4f}")
        print("   -> 后续输出将直接显示真实的 Log10 亮度。")
    else:
        print("   -> 警告：未找到校准数据，使用原始预测")
        lr = None
    # ============================================================
    
    print("\n" + "="*60)
    print("🧬 GFP 亮度预测器已就绪！(物理数值对齐版)")
    print("请输入氨基酸序列 (输入 'q' 退出):")
    print("="*60)
    
    while True:
        seq = input("\n👉 请粘贴序列: ").strip().upper()
        
        if seq.lower() == 'q':
            print("退出测试。"); break
            
        if not seq or not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
            print("⚠️ 序列无效。"); continue
            
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=250).to(device)
        with torch.no_grad():
            raw_pred_log = model(inputs["input_ids"], inputs["attention_mask"]).item()
            
        # 🌟 线性回归校准
        if lr is not None:
            calibrated_log = lr.predict([[raw_pred_log]])[0]
        else:
            calibrated_log = raw_pred_log
        
        # 计算相对 WT 的倍数 (以 WT=3.7192 为基准)
        TRUE_WT_LOG = 3.7192
        relative_brightness = 10 ** (calibrated_log - TRUE_WT_LOG)
        
        print("-" * 40)
        print(f"📊 预测真实 Log10 亮度: {calibrated_log:.4f}  (对应绝对亮度: {10**calibrated_log:.0f})")
        print(f"💡 预测相对 WT 亮度:   {relative_brightness:.3f} 倍")
        
        if relative_brightness >= 1.5:
            print("🔥 评价: 显著超亮！(极具潜力的六边形战士)")
        elif relative_brightness >= 0.9:
            print("✅ 评价: 亮度正常 (与 WT 相当，Log10 约 3.7)")
        elif relative_brightness >= 0.1:
            print("⚠️ 评价: 亮度受损 (可能存在折叠缺陷)")
        else:
            print("💀 评价: 死蛋白 (亮度接近仪器底噪，Log10 约 1.3)")
        print("-" * 40)

if __name__ == "__main__":
    main()
