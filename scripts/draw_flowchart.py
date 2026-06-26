"""
生成项目流程图 (科研级PNG图片) - v2
修正: 加大字体, 调整框与文字比例, 防止溢出
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 整体画布更大, 给足空间
fig, ax = plt.subplots(1, 1, figsize=(16, 22), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis('off')

# 颜色方案
COLOR_PHASE = '#2E74B5'
COLOR_BOX = '#D6E4F0'
COLOR_BOX_EDGE = '#2E74B5'
COLOR_ARROW = '#2E74B5'
COLOR_DETAIL = '#444444'
COLOR_HIGHLIGHT = '#FFC000'

def draw_phase_box(ax, x, y, w, h, title, details, phase_num=None):
    """绘制一个Phase框"""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=COLOR_BOX, edgecolor=COLOR_BOX_EDGE,
                          linewidth=2.5)
    ax.add_patch(box)

    # Phase编号标签 (左上角)
    if phase_num is not None:
        badge_w, badge_h = 1.8, 0.55
        badge_x = x + 0.2
        badge_y = y + h - badge_h - 0.15
        badge = FancyBboxPatch((badge_x, badge_y), badge_w, badge_h,
                                boxstyle="round,pad=0.08",
                                facecolor=COLOR_PHASE, edgecolor=COLOR_PHASE)
        ax.add_patch(badge)
        ax.text(badge_x + badge_w/2, badge_y + badge_h/2,
                f'Phase {phase_num}', fontsize=13, fontweight='bold',
                color='white', ha='center', va='center')
        title_x = badge_x + badge_w + 0.3
    else:
        title_x = x + 0.4

    # 标题
    title_y = y + h - 0.55
    ax.text(title_x, title_y, title, fontsize=14, fontweight='bold',
            color=COLOR_PHASE, ha='left', va='center')

    # 细节文字
    for i, line in enumerate(details):
        ax.text(x + 0.5, title_y - 0.65 - i*0.5, line,
                fontsize=11, color=COLOR_DETAIL, ha='left', va='center')

def draw_arrow(ax, x, y_start, y_end):
    """绘制向下的箭头"""
    ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle='->', color=COLOR_ARROW,
                               lw=3, connectionstyle='arc3,rad=0'))

def draw_side_note(ax, x, y, lines, color=COLOR_HIGHLIGHT):
    """绘制侧边注释"""
    text = '\n'.join(lines)
    ax.text(x, y, text, fontsize=11, color='#333333',
            ha='left', va='center', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                     edgecolor='#D4A017', alpha=0.85, linewidth=1.5))

# ======================== 绘制流程图 ========================

# 标题
ax.text(8, 21.3, 'GFP亮度优化设计 — 项目整体流程', fontsize=20, fontweight='bold',
        color='#1A1A1A', ha='center', va='center')
ax.text(8, 20.8, 'GFP Hexagon Warrior — SynBio Challenges 2025', fontsize=12,
        color='#666666', ha='center', va='center')

# Phase 1
draw_phase_box(ax, 1, 18.2, 11.5, 2.0,
               '数据清洗与模型训练',
               ['141,572条实验数据 → 清洗去重(~137k) → 72k优质筛选',
                'ESM-2 LoRA微调 + Frozen MLP → V5/V7/V9三模型'],
               phase_num=1)
draw_side_note(ax, 13, 19.2,
               ['三模型加权:', 'V9(50%) + V7(35%)', '+ V5(15%)'])

draw_arrow(ax, 6.75, 18.2, 17.5)

# Phase 2
draw_phase_box(ax, 1, 15.5, 11.5, 2.0,
               '序列生成: 高亮度父本 × 随机突变',
               ['各GFP类型Top 1000父本, 每条20变体(1-5个突变)',
                '保护发色团(TYG)及核心区 → V5粗筛 → 32,394条'],
               phase_num=2)

draw_arrow(ax, 6.75, 15.5, 14.8)

# Phase 2.5
draw_phase_box(ax, 1, 12.2, 11.5, 2.6,
               '分层突变 + 三模型共识筛选 (5轮迭代)',
               ['各类型Top 50父本 × 500变体/条, 固定3个突变',
                'V7初筛(各类型独立阈值)',
                'V5+V7+V9共识投票(≥2票通过)',
                '产出: 6,626条V7过滤 + 交叉验证1,597条'],
               phase_num='2.5')
draw_side_note(ax, 13, 13.5,
               ['加权公式:', '0.50×V9', '+ 0.35×V7', '+ 0.15×V5'])

draw_arrow(ax, 6.75, 12.2, 11.5)

# Phase 3
draw_phase_box(ax, 1, 9.8, 11.5, 1.7,
               'ESM3-ddG 热稳定性打分',
               ['预测突变自由能变化(ddG): <0稳定化, >0去稳定化',
                '仅辅助过滤, 不参与亮度排名 (r=0.175)'],
               phase_num=3)

draw_arrow(ax, 6.75, 9.8, 9.1)

# Phase 4
draw_phase_box(ax, 1, 7.2, 11.5, 1.8,
               '综合筛选 → Top 20',
               ['Exclusion List过滤(135,414条已知序列)',
                '综合评分排名 → 4种GFP类型各取Top → 多样性配额'],
               phase_num=4)

draw_arrow(ax, 6.75, 7.2, 6.5)

# Phase 5
draw_phase_box(ax, 1, 3.2, 11.5, 3.2,
               '最终候选选取与人工审核',
               ['四数据源合并去重(39,020条) → V5/V7/V9统一打分',
                '4类型各≥1条 + 第5条给最高分类型 → 选5条候选',
                'AlphaFold 3结构验证: beta-barrel完整性',
                '发色团pLDDT≥70, 确保72°C热处理下结构稳定',
                '最终提交6条序列'],
               phase_num=5)
draw_side_note(ax, 13, 4.8,
               ['参考标准:', 'mBaoJin GFP', '(最亮单体荧光蛋白)'])

# 底部总结
ax.text(8, 2.2, '数据流: 141,572条 → 137k有效 → 72k优质 → 100万+生成 → 39,020去重 → 5条最终候选',
        fontsize=12, color='#888888', ha='center', va='center', style='italic')

plt.tight_layout()
out_path = str(Path(__file__).resolve().parent.parent / 'assets' / 'project_flowchart.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'流程图已保存: {out_path}')
plt.close()
