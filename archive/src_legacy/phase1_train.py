"""
Phase 1.3: 终极模型训练 (硬件加速优化版)
强制 XGBoost 使用 GPU，明确打印各模型的硬件调用状态。
"""
import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from src.config import (FEATS_DIR, MODELS_DIR, RANDOM_SEED, AUTHOR, DATE_STR, seed_everything, get_device)

# ================= 1. 高级 PyTorch MLP 定义 =================
class AdvancedBrightnessMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(AdvancedBrightnessMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.network(x)

# ================= 2. 单模型训练函数 =================
def train_xgboost(X_tr, Y_tr, X_vl, Y_vl, use_gpu: bool):
    device_str = "cuda" if use_gpu else "cpu"
    print(f"   🌲 XGBoost 正在使用设备: [{device_str.upper()}]")
    
    model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=8, learning_rate=0.05, 
        objective='reg:squarederror', random_state=RANDOM_SEED, 
        tree_method='hist', 
        device=device_str,  # 🌟 XGBoost 2.0+ 强制指定 GPU
        early_stopping_rounds=50
    )
    model.fit(X_tr, Y_tr, eval_set=[(X_vl, Y_vl)], verbose=0)
    return model

def train_lightgbm(X_tr, Y_tr, X_vl, Y_vl):
    # LightGBM 在 Windows 下默认 CPU，但 7950X3D 跑这个极快，无需强求 GPU
    print(f"   🌳 LightGBM 正在使用设备: [CPU] (7950X3D 多核加速中...)")
    model = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=8, learning_rate=0.05, 
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )
    model.fit(X_tr, Y_tr, eval_set=[(X_vl, Y_vl)], 
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return model

def train_mlp(X_tr, Y_tr, X_vl, Y_vl, device):
    print(f"   🧠 PyTorch MLP 正在使用设备: [{str(device).upper()}]")
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32).view(-1, 1).to(device)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32).to(device)
    Y_vl_t = torch.tensor(Y_vl, dtype=torch.float32).view(-1, 1).to(device)
    
    model = AdvancedBrightnessMLP(X_tr_t.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()
    
    best_val_r2, best_state, trigger = -np.inf, None, 0
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 512):
            b_idx = idx[i:i+512]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[b_idx]), Y_tr_t[b_idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_r2 = r2_score(Y_vl_t.cpu().numpy(), model(X_vl_t).cpu().numpy())
        if val_r2 > best_val_r2:
            best_val_r2, best_state, trigger = val_r2, model.state_dict(), 0
        else:
            trigger += 1
        if trigger >= 15: break
            
    model.load_state_dict(best_state)
    return model

# ================= 3. 主执行逻辑 (含 Stacking) =================
def train_models(force: bool = False):
    seed_everything()
    device = get_device()
    use_gpu = (device.type == 'cuda')
    
    print(f"\n🚀 [Phase 1.3] 终极模型训练 (硬件加速优化版)")
    print(f"💻 检测到 PyTorch 主设备: {device}")
    
    existing = list(MODELS_DIR.glob(f"GFP_Stacking_*_{AUTHOR}_*.joblib"))
    if existing and not force:
        print("⚠️ 检测到已存在 Stacking 模型。如需重跑请加 --force")
        return

    feat_files = list(FEATS_DIR.glob("GFP_Feat_Full_ESM2t33_*.joblib"))
    if not feat_files: raise FileNotFoundError("❌ 未找到特征！")
    data = joblib.load(max(feat_files, key=lambda p: p.stat().st_mtime))
    X, Y = data['X'], data['Y']
    
    # 划分数据集
    X_tv, X_test, Y_tv, Y_test = train_test_split(X, Y, test_size=0.1, random_state=RANDOM_SEED)
    X_train, X_val, Y_train, Y_val = train_test_split(X_tv, Y_tv, test_size=0.111, random_state=RANDOM_SEED)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # 1. 训练 Base Models
    print("\n-> [1/4] 训练 Base Models...")
    xgb_m = train_xgboost(X_train_s, Y_train, X_val_s, Y_val, use_gpu)
    lgb_m = train_lightgbm(X_train_s, Y_train, X_val_s, Y_val)
    mlp_m = train_mlp(X_train_s, Y_train, X_val_s, Y_val, device)
    
    # 2. 生成 Stacking 特征
    print("\n-> [2/4] 生成 Stacking 元特征...")
    meta_val = np.column_stack([
        xgb_m.predict(X_val_s),
        lgb_m.predict(X_val_s),
        mlp_m(torch.tensor(X_val_s, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten()
    ])
    meta_test = np.column_stack([
        xgb_m.predict(X_test_s),
        lgb_m.predict(X_test_s),
        mlp_m(torch.tensor(X_test_s, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten()
    ])
    
    # 3. 训练 Meta-learner
    print("-> [3/4] 训练 Meta-learner (Ridge Regression on CPU)...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_val, Y_val)
    
    # 4. 最终评估
    print("-> [4/4] 评估 Test Set...")
    xgb_r2 = r2_score(Y_test, xgb_m.predict(X_test_s))
    lgb_r2 = r2_score(Y_test, lgb_m.predict(X_test_s))
    mlp_r2 = r2_score(Y_test, mlp_m(torch.tensor(X_test_s, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten())
    stack_pred = meta_model.predict(meta_test)
    stack_r2 = r2_score(Y_test, stack_pred)
    stack_rmse = np.sqrt(mean_squared_error(Y_test, stack_pred))
    
    print("\n" + "="*50)
    print("🏆 模型 PK 结果 (Test Set R2):")
    print(f"   XGBoost (GPU):  {xgb_r2:.4f}")
    print(f"   LightGBM (CPU): {lgb_r2:.4f}")
    print(f"   MLP (GPU):      {mlp_r2:.4f}")
    print(f"   👑 Stacking:    {stack_r2:.4f} (RMSE: {stack_rmse:.4f})")
    print("="*50)
    
    # 5. 保存 Stacking 全家桶
    out_dict = {
        "meta_model": meta_model,
        "base_models": {"xgb": xgb_m, "lgbm": lgb_m, "mlp": mlp_m},
        "scaler": scaler,
        "metrics": {"stacking_r2": stack_r2, "stacking_rmse": stack_rmse, 
                    "xgb_r2": xgb_r2, "lgbm_r2": lgb_r2, "mlp_r2": mlp_r2},
        "model_type": "Stacking",
        "provenance": {"dataset_version": max(feat_files, key=lambda p: p.stat().st_mtime).name},
        "metadata": {"train_date": DATE_STR, "author": AUTHOR}
    }
    
    out_path = MODELS_DIR / f"GFP_Stacking_ESM2t33_{AUTHOR}_{DATE_STR}_v6_ultimate.joblib"
    joblib.dump(out_dict, out_path)
    print(f"💾 Stacking 模型全家桶已保存至: {out_path}\n")
