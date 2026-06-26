"""分析20条湿实验序列的排名能力"""
import re
import numpy as np

# 读取完整输出
with open(r'C:\Users\Type002\.qoder-cn\cache\projects\GFP_Hexagon_Warrior-77fb15bc\agent-tools\0c7fad29\9b8b2a55.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 解析每个序列各类型的预测
seq_data = []
current_seq = None
for line in content.split('\n'):
    line = line.strip()
    m = re.match(r'#(\d+)\s+\(year=(\d+)\)', line)
    if m:
        current_seq = {'idx': int(m.group(1)), 'year': int(m.group(2))}
        continue
    if current_seq is not None:
        for gt in ['avGFP', 'amacGFP', 'cgreGFP', 'ppluGFP']:
            if gt in line:
                nums = re.findall(r'(\d+\.\d+)', line)
                if len(nums) >= 4:
                    current_seq[f'{gt}_v5'] = float(nums[0])
                    current_seq[f'{gt}_v7'] = float(nums[1])
                    current_seq[f'{gt}_v9'] = float(nums[2])
                    current_seq[f'{gt}_avg'] = float(nums[3])
        if '>>>' in line and 'ppluGFP' in line or '>>>' in line and 'cgreGFP' in line:
            m2 = re.search(r'(\d+\.\d+)$', line)
            if m2:
                current_seq['best_avg'] = float(m2.group(1))
                seq_data.append(current_seq)
                current_seq = None

print(f"解析到 {len(seq_data)} 条序列")
print(f"\n{'='*70}")
print(f" 方案3验证: 模型排名能力")
print(f"{'='*70}")

# 按avGFP加权平均排序
print(f"\n=== 作为avGFP的预测排名 ===")
print(f"  {'#':>3} {'突变':>4} {'V5':>6} {'V7':>6} {'V9':>6} {'加权':>6}")
avGFP_avgs = [(d['idx'], d.get('avGFP_v5',0), d.get('avGFP_v7',0), d.get('avGFP_v9',0), d.get('avGFP_avg',0)) for d in seq_data]
avGFP_avgs.sort(key=lambda x: x[4], reverse=True)
for rank, (idx, v5, v7, v9, avg) in enumerate(avGFP_avgs, 1):
    print(f"  {rank:3d}  #{idx:<2d}    {v5:6.3f} {v7:6.3f} {v9:6.3f} {avg:6.3f}")

# 按最佳类型加权平均排序
print(f"\n=== 最佳类型加权平均排名 ===")
print(f"  {'排名':>4} {'#':>3} {'最佳类型':>8} {'加权平均':>8}")
ranked = sorted(seq_data, key=lambda x: x.get('best_avg', 0), reverse=True)
for rank, d in enumerate(ranked, 1):
    best_type = 'ppluGFP' if d.get('ppluGFP_avg',0) >= d.get('cgreGFP_avg',0) else 'cgreGFP'
    print(f"  {rank:4d}  #{d['idx']:<2d}  {best_type:>8}  {d.get('best_avg',0):8.3f}")

# 关键问题: 排名是否有意义?
print(f"\n{'='*70}")
print(f" 排名能力分析")
print(f"{'='*70}")

avGFP_preds = [d.get('avGFP_v7', 0) for d in seq_data]
print(f"\n  作为avGFP的V7预测分布:")
print(f"    范围: [{min(avGFP_preds):.3f}, {max(avGFP_preds):.3f}]")
print(f"    >3.5: {sum(1 for x in avGFP_preds if x>3.5)}/20")
print(f"    >3.0: {sum(1 for x in avGFP_preds if x>3.0)}/20")
print(f"    >2.0: {sum(1 for x in avGFP_preds if x>2.0)}/20")
print(f"    >1.5: {sum(1 for x in avGFP_preds if x>1.5)}/20")

print(f"\n  问题: 20条序列都是实测>4.0的亮序列")
print(f"  但模型预测(avGFP)最高仅 {max(avGFP_preds):.3f}")
print(f"  排名虽然能区分高低, 但所有值都严重偏低")
print(f"  如果生成1万条变体, 模型可能把真正的>4.0序列排在")
print(f"  预测值3.5的位置, 而把真正只有3.7的序列排在3.0")
print(f"  → 排名仍然有用, 但需要用实验验证Top N")
