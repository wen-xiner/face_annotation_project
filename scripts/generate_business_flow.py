"""
生成系统整体业务场景示意图（大字体版v3）
数据采集 -> 自动标注 -> 审核修正 -> 格式导出
配色方案B：青绿科技
重点：子项字体要大，卡片内部充分利用空间，不重叠
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

# ===== 中文字体 =====
fm._load_fontmanager(try_read_cache=False)
font_path = None
for f in fm.fontManager.ttflist:
    if 'Microsoft YaHei' in f.name and f.fname.endswith('.ttc'):
        font_path = f.fname; break
if not font_path:
    for f in fm.fontManager.ttflist:
        if 'msyh' in f.fname.lower():
            font_path = f.fname; break

if font_path:
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
else:
    font_name = 'SimHei'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
CN_FONT = fm.FontProperties(fname=font_path) if font_path else fm.FontProperties(family=font_name)

# ===== 配色方案B =====
C = {
    'collect': '#3F51B5', 'annotate': '#009688', 'review': '#26A69A', 'export': '#00695C',
    'arrow': '#FFC107', 'arrow_dim': '#FFE082', 'bg': '#FAFAFA', 'card_bg': '#FFFFFF',
    'title_color': '#212121', 'subtitle': '#616161', 'text': '#424242', 'text_light': '#757575',
    'border': '#E0E0E0', 'shadow': '#E8E8E8',
}

# ===== 阶段定义 =====
STAGES = [
    {
        'id': '01', 'name': '数据采集', 'color': C['collect'], 'color_light': '#C5CAE9',
        'items': ['本地文件拖拽上传', '摄像头拍照与录像', 'URL远程链接下载', '外部文件夹批量导入'],
        'output_label': '原始图片集', 'output_path': 'dataset/raw_images/',
    },
    {
        'id': '02', 'name': '自动标注', 'color': C['annotate'], 'color_light': '#B2DFDB',
        'items': ['MediaPipe BlazeFace 人脸检测', '478稠密点简化5点关键点', '结构化JSON标注生成', '三维质量评分 (模糊/亮度/姿态)'],
        'output_label': '自动标注JSON', 'output_path': 'output/annotations/auto_labeled/',
    },
    {
        'id': '03', 'name': '审核修正', 'color': C['review'], 'color_light': '#B2DFDB',
        'items': ['JSON结构完整性校验', '三维内容质量评估', '人工交互逐字段修正', '几何约束自动纠错'],
        'output_label': '审核后JSON', 'output_path': 'output/annotations/manually_reviewed/',
    },
    {
        'id': '04', 'name': '格式导出', 'color': C['export'], 'color_light': '#C8E6C9',
        'items': ['COCO JSON 格式导出', 'YOLO txt 格式导出', 'Pascal VOC XML 格式导出', 'CSV 表格格式导出'],
        'output_label': '最终交付物', 'output_path': 'output/exports/',
    },
]

# ===== 画布 =====
FIG_W, FIG_H = 28, 12.5
DPI = 300
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C['bg'])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect('equal')
ax.axis('off')

# ===== 布局 =====
CARD_W = 5.2
CARD_H = 6.8
CARD_Y_BOTTOM = 3.0
TITLE_BAR_H = 0.90
CORNER_R = 0.16

total_w = 4 * CARD_W
gap = (FIG_W - total_w) / 5
card_xs = [gap + i * (CARD_W + gap) for i in range(4)]
arrow_y = CARD_Y_BOTTOM + CARD_H / 2

# ===== 字号（重点：子项要大） =====
F_TITLE      = 34
F_SUBTITLE   = 14
F_STAGE_ID   = 20
F_STAGE_NAME = 22
F_ITEM       = 17
F_TAG        = 13
F_PATH       = 10
F_ARROW_LABEL = 11
F_PIPE       = 11
F_LEGEND     = 11


def add_box(ax, x, y, w, h, r, fc, ec, lw=1.5, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        mutation_scale=w,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    ))


def add_shadow(ax, x, y, w, h, r):
    ax.add_patch(FancyBboxPatch(
        (x + 0.10, y - 0.10), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        mutation_scale=w,
        facecolor=C['shadow'], edgecolor='none', linewidth=0, zorder=0,
    ))


# ===== 主标题区 =====
title_y = FIG_H - 0.8
ax.text(FIG_W / 2, title_y, '系统整体业务场景示意图',
        fontsize=F_TITLE, fontweight='bold', color=C['title_color'],
        ha='center', va='top', fontproperties=CN_FONT)

subtitle_y = FIG_H - 1.35
ax.text(FIG_W / 2, subtitle_y,
        'System Business Scenario: Collection  ->  Annotation  ->  Review  ->  Export',
        fontsize=F_SUBTITLE, color=C['subtitle'], ha='center', va='top', style='italic')

# 装饰线
lx = FIG_W / 2
ax.plot([lx - 5, lx + 5], [FIG_H - 1.65, FIG_H - 1.65],
        color=C['arrow'], lw=3, solid_capstyle='round', zorder=5)
ax.plot(lx - 5, FIG_H - 1.65, 'o', color=C['arrow'], markersize=7, zorder=5)
ax.plot(lx + 5, FIG_H - 1.65, 'o', color=C['arrow'], markersize=7, zorder=5)


# ===== 绘制4个卡片 =====
for i, stage in enumerate(STAGES):
    cx = card_xs[i]
    cy = CARD_Y_BOTTOM

    # 阴影
    add_shadow(ax, cx, cy, CARD_W, CARD_H, CORNER_R)

    # 卡片主体
    add_box(ax, cx, cy, CARD_W, CARD_H, CORNER_R, C['card_bg'], C['border'], lw=1.2, zorder=1)

    # 彩色标题栏
    tbb = cy + CARD_H - TITLE_BAR_H  # title_bar_bottom
    add_box(ax, cx, tbb, CARD_W, TITLE_BAR_H + CORNER_R, CORNER_R,
            stage['color'], 'none', lw=0, zorder=2)

    # 覆盖标题栏底部多余的圆角
    ax.add_patch(plt.Rectangle(
        (cx + 0.01, tbb), CARD_W - 0.02, CORNER_R + 0.03,
        facecolor=C['card_bg'], edgecolor='none', zorder=3))

    # 编号圆圈
    cr = 0.50
    ccx = cx + 0.65
    ccy = tbb + TITLE_BAR_H / 2
    ax.add_patch(Circle((ccx, ccy), cr, facecolor='white',
                 edgecolor='white', linewidth=3.5, zorder=4, alpha=0.95))
    ax.text(ccx, ccy, stage['id'], fontsize=F_STAGE_ID, fontweight='bold',
            color=stage['color'], ha='center', va='center', zorder=5)

    # 阶段名称
    ax.text(cx + 1.40, ccy, stage['name'], fontsize=F_STAGE_NAME,
            fontweight='bold', color='white', ha='left', va='center',
            fontproperties=CN_FONT, zorder=5)

    # ---- 子项列表（卡片核心内容，字号要大） ----
    item_y_start = tbb - 0.40
    item_spacing = 1.12       # 4个子项在约4.5英寸高度内均匀分布

    for j, item_text in enumerate(stage['items']):
        item_y = item_y_start - j * item_spacing

        # 彩色圆点
        dot_r = 0.13
        ax.add_patch(Circle((cx + 0.42, item_y), dot_r,
                     facecolor=stage['color'], edgecolor='none', zorder=4, alpha=0.85))

        # 子项文字（大字号）
        ax.text(cx + 0.75, item_y, item_text,
                fontsize=F_ITEM, color=C['text'],
                ha='left', va='center', fontproperties=CN_FONT, zorder=4)

    # ---- 底部输出标签 ----
    tag_y = cy + 0.70
    tag_h = 0.50
    tag_rect = FancyBboxPatch(
        (cx + 0.30, tag_y - tag_h/2), CARD_W - 0.60, tag_h,
        boxstyle="round,pad=0.01,rounding_size=0.12",
        mutation_scale=CARD_W,
        facecolor=stage['color_light'], edgecolor='none',
        linewidth=0, zorder=3, alpha=0.45,
    )
    ax.add_patch(tag_rect)

    ax.text(cx + CARD_W / 2, tag_y, f'> {stage["output_label"]}',
            fontsize=F_TAG, color=stage['color'], ha='center', va='center',
            fontproperties=CN_FONT, fontweight='bold', zorder=4)

    ax.text(cx + CARD_W / 2, tag_y - tag_h/2 - 0.38, stage['output_path'],
            fontsize=F_PATH, color=C['text_light'], ha='center', va='center',
            style='italic', zorder=4)


def mcolors_to_rgb(hex_color):
    """把 hex 颜色转成 (r,g,b) 0-1 范围"""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


# ===== 阶段间箭头 =====
for i in range(3):
    x1 = card_xs[i] + CARD_W
    x2 = card_xs[i + 1]
    ax.annotate(
        '', xy=(x2 - 0.36, arrow_y), xytext=(x1 + 0.18, arrow_y),
        arrowprops=dict(
            arrowstyle='->,head_width=0.7,head_length=0.55',
            color=C['arrow'], lw=5.0, shrinkA=0, shrinkB=0,
            connectionstyle='arc3,rad=0',
        ), zorder=6,
    )
    labels = ['原始图片', '自动标注JSON', '审核后JSON']
    ax.text((x1 + x2) / 2, arrow_y - 0.72, labels[i],
            fontsize=F_ARROW_LABEL, color=C['text_light'],
            ha='center', va='top', fontproperties=CN_FONT, style='italic', zorder=5)


# ===== 底部全流程管道线 =====
pipe_y = 1.4
pipe_left = card_xs[0] + 0.5
pipe_right = card_xs[3] + CARD_W - 0.5

ax.plot([pipe_left, pipe_right], [pipe_y, pipe_y],
        color=C['arrow'], lw=3.5, alpha=0.30, zorder=2, solid_capstyle='round')
ax.text((pipe_left + pipe_right) / 2, pipe_y - 0.55,
        '全流程数据处理管道 (Full-Pipeline Data Processing)',
        fontsize=F_PIPE, color=C['text_light'], ha='center', va='top',
        fontproperties=CN_FONT, style='italic', zorder=3)

for i in range(4):
    cx_c = card_xs[i] + CARD_W / 2
    ax.plot([cx_c, cx_c], [CARD_Y_BOTTOM, pipe_y],
            color=C['arrow_dim'], lw=1.5, ls='--', alpha=0.45, zorder=1)
    ax.plot(cx_c, pipe_y, 'o', color=C['arrow'], markersize=6, zorder=3)


# ===== 图注 =====
legend_y = 0.55
legend_items = [
    ('数据采集', C['collect']), ('自动标注', C['annotate']),
    ('审核修正', C['review']), ('格式导出', C['export']),
]
lx0 = FIG_W / 2 - 4.6
for i, (label, color) in enumerate(legend_items):
    lx = lx0 + i * 2.5
    ax.plot(lx, legend_y, 's', color=color, markersize=12, zorder=5)
    ax.text(lx + 0.35, legend_y, label, fontsize=F_LEGEND, color=C['text'],
            ha='left', va='center', fontproperties=CN_FONT, zorder=5)


# ===== 保存 =====
output_dir = r'E:\face_annotation_project\output\charts'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '图2-1_业务场景示意图.png')

fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor=C['bg'],
            edgecolor='none', pad_inches=0.3)
plt.close(fig)
print(f'[OK] {output_path}')
print(f'     {FIG_W}x{FIG_H}in @ {DPI}dpi = ~{FIG_W*DPI}x{FIG_H*DPI}px')
