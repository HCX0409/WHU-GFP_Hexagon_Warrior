"""
V9 LoRA 亮度预测工具
输入: 序列 + GFP类型 → 输出: 预测亮度 (delta + WT基线)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from src.config import LOCAL_ESM2_DIR, MODELS_DIR

GFP_TYPE_LIST = ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']
GFP_TYPE_TO_IDX = {t: i for i, t in enumerate(GFP_TYPE_LIST)}
WT_BRIGHTNESS = {'avGFP': 3.7192, 'amacGFP': 3.9707, 'cgreGFP': 4.4969, 'ppluGFP': 4.2258}

MODEL_DIR = MODELS_DIR / "ESM2_LoRA_V9_72k_HCX_260618"


class ESM2LoRARegressorV9(nn.Module):
    def __init__(self, base_model, num_gfp_types=4):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
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
        pooled = self.pool_norm(pooled)
        type_onehot = torch.zeros(pooled.size(0), self.num_gfp_types, device=pooled.device)
        type_onehot.scatter_(1, gfp_type_idx.unsqueeze(1), 1.0)
        combined = torch.cat([pooled, type_onehot], dim=1)
        return self.regressor(combined).squeeze(-1)


def load_model():
    """加载V9模型 (LoRA已merge)"""
    print("加载V9 LoRA模型...")
    model_name = str(LOCAL_ESM2_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    peft_model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
    peft_model = peft_model.merge_and_unload()

    model = ESM2LoRARegressorV9(peft_model, num_gfp_types=4)
    regressor_state = torch.load(MODEL_DIR / "regressor.pt", map_location='cpu', weights_only=True)

    full_state = {}
    for k, v in regressor_state.items():
        full_state[f'regressor.{k}'] = v
    model_state = model.state_dict()
    for k in ['pool_norm.weight', 'pool_norm.bias']:
        if k in model_state:
            full_state[k] = model_state[k]

    model.load_state_dict(full_state, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"  模型加载完成 (device: {device})")
    return model, tokenizer, device


@torch.no_grad()
def predict_brightness(model, tokenizer, device, sequence, gfp_type):
    """
    预测单条序列的亮度
    返回: (delta, predicted_brightness, wt_brightness)
    """
    assert gfp_type in WT_BRIGHTNESS, f"未知GFP类型: {gfp_type}, 可选: {list(WT_BRIGHTNESS.keys())}"

    enc = tokenizer(sequence, truncation=True, max_length=250, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    gfp_type_idx = torch.tensor([GFP_TYPE_TO_IDX[gfp_type]], dtype=torch.long, device=device)

    with torch.amp.autocast('cuda'):
        delta = model(input_ids, attention_mask, gfp_type_idx).cpu().float().item()

    wt = WT_BRIGHTNESS[gfp_type]
    predicted_brightness = delta + wt

    return delta, predicted_brightness, wt


def main():
    model, tokenizer, device = load_model()

    print(f"\n{'='*60}")
    print("  V9 LoRA 亮度预测工具")
    print(f"  可选GFP类型: {GFP_TYPE_LIST}")
    print(f"  WT基线: {WT_BRIGHTNESS}")
    print(f"  预测公式: 亮度 = delta + WT基线")
    print(f"{'='*60}")
    print("  输入 'q' 退出\n")

    while True:
        try:
            seq = input("序列 (氨基酸, 如 MVLSPAD...): ").strip()
            if seq.lower() == 'q':
                break
            if not seq:
                continue

            gfp_type = input(f"GFP类型 {GFP_TYPE_LIST}: ").strip()
            if gfp_type.lower() == 'q':
                break
            if gfp_type not in GFP_TYPE_LIST:
                print(f"  ❌ 未知类型 '{gfp_type}', 请从 {GFP_TYPE_LIST} 中选择")
                continue

            delta, pred, wt = predict_brightness(model, tokenizer, device, seq, gfp_type)

            print(f"\n  ┌─────────────────────────────────────")
            print(f"  │ GFP类型:  {gfp_type}")
            print(f"  │ WT基线:   {wt:.4f}")
            print(f"  │ Delta:    {delta:+.4f}")
            print(f"  │ 预测亮度: {pred:.4f}")
            print(f"  └─────────────────────────────────────\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  ❌ 错误: {e}\n")

    print("再见!")


if __name__ == "__main__":
    main()
