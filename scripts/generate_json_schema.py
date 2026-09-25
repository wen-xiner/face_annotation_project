"""
标注JSON字段结构示意图
必需字段5个 + 可选字段6个，层次关系
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import os

# ===== 字体 =====
fm._load_fontmanager(try_read_cache=False)
fp = None
for f in fm.fontManager.ttflist:
    if 'Microsoft YaHei' in f.name and f.fname.endswith('.ttc'):
        fp = f.fname; break
if not fp:
    for f in fm.fontManager.ttflist:
        if 'msyh' in f.fname.lower(): fp = f.fname; break
if fp:
    fm.fontManager.addfont(fp)
    fn = fm.FontProperties(fname=fp).get_name()
else:
    fn = 'SimHei'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [fn, 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
CF = fm.FontProperties(fname=fp) if fp else fm.FontProperties(family=fn)

# ===== 配色 =====
C_REQ = '#3F51B5'      # 必需字段 - 靛蓝
C_OPT = '#FFC107'      # 可选字段 - 琥珀
C_ROOT = '#009688'     # 根节点 - 青绿
C_SUB = '#26A69A'      # 子对象 - 浅青
BG = '#FFFFFF'; TX = '#212121'; LT = '#888888'

# ===== 画布 =====
FW, FH = 32, 18
DPI = 300
fig = plt.figure(figsize=(FW, FH), facecolor=BG)
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0,FW); ax.set_ylim(0,FH); ax.set_aspect('equal'); ax.axis('off')

# ===== 标题 =====
ax.text(FW/2, FH-0.7, '标注JSON字段结构示意图', fontsize=38, fontweight='bold',
        color=TX, ha='center', va='top', fontproperties=CF)
ax.text(FW/2, FH-1.25, 'Annotation JSON Schema — 5 Required + Optional Fields',
        fontsize=15, color=LT, ha='center', va='top', style='italic')
ax.plot([FW/2-3, FW/2+3], [FH-1.5, FH-1.5], color=C_OPT, lw=2.5, solid_capstyle='round', zorder=5)

# ===== 辅助 =====
def rbox(ax, x, y, w, h, r, fc, ec='none', lw=0, z=1, a=1):
    ax.add_patch(FancyBboxPatch((x,y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={r}", mutation_scale=max(w,1),
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=a))

def field_card(ax, x, y, w, h, name, type_str, desc, color, is_required=True):
    """绘制字段卡片"""
    # 卡片背景
    rbox(ax, x, y, w, h, 0.08, color, z=3, a=0.10)
    # 顶部色条
    rbox(ax, x, y+h-0.32, w, 0.32, 0.08, color, z=4, a=0.5)
    # 标签
    tag = '[必填]' if is_required else '[可选]'
    tag_color = color
    ax.text(x+0.15, y+h-0.16, f'{tag} {name}', fontsize=15, fontweight='bold',
            color=color, ha='left', va='center', fontproperties=CF, zorder=5)
    # 类型
    ax.text(x+0.15, y+h-0.52, f'类型: {type_str}', fontsize=11, color=LT,
            ha='left', va='center', zorder=5)
    # 描述
    ax.text(x+0.15, y+0.35, desc, fontsize=10.5, color=TX,
            ha='left', va='center', fontproperties=CF, zorder=5, linespacing=1.3)

def sub_field(ax, x, y, name, type_str, color):
    """子字段小标签"""
    ax.text(x, y, f'{name}: {type_str}', fontsize=9.5, color=color,
            ha='left', va='center', fontproperties=CF, zorder=5)

def conn_line(ax, x0, y0, x1, y1, color=LT):
    """连接线"""
    ax.plot([x0, x1], [y0, y1], color=color, lw=1.2, zorder=2)

def conn_branch(ax, px, py, children_x, children_y, color=LT):
    """分支连接：从父节点向下分叉到多个子节点"""
    # 先画垂直主干到分叉点
    branch_y = py - 0.25
    ax.plot([px, px], [py, branch_y], color=color, lw=1.5, zorder=2)
    # 水平横线
    x_min = min(children_x)
    x_max = max(children_x)
    ax.plot([x_min, x_max], [branch_y, branch_y], color=color, lw=1.5, zorder=2)
    # 到每个子节点的垂直线
    for cx, cy in zip(children_x, children_y):
        ax.plot([cx, cx], [branch_y, cy], color=color, lw=1.2, zorder=2)


# ===== 布局 =====
# 根节点
ROOT_X = FW/2 - 2.5; ROOT_Y = FH - 4.0
ROOT_W, ROOT_H = 5.0, 1.0

# 两列：左=必需字段(5个)，右=可选字段(6个)
LEFT_X = 3.0; RIGHT_X = FW/2 + 0.5
CARD_W, CARD_H = 12.5, 1.65
ROW_GAP = 0.22

# ===== 根节点 =====
rbox(ax, ROOT_X, ROOT_Y, ROOT_W, ROOT_H, 0.12, C_ROOT, z=4, a=0.20)
ax.text(ROOT_X+ROOT_W/2, ROOT_Y+ROOT_H/2, 'Annotation JSON', fontsize=22, fontweight='bold',
        color=C_ROOT, ha='center', va='center', zorder=5)
# JSON Schema 标注
ax.text(ROOT_X+ROOT_W/2, ROOT_Y-0.3, '基于 JSON Schema Draft 2020-12',
        fontsize=10, color=LT, ha='center', va='top', style='italic', zorder=3)

# ===== 左列标题：必需字段 =====
lbl_y = ROOT_Y - 1.0
rbox(ax, LEFT_X-0.1, lbl_y-0.35, 2.2, 0.7, 0.08, C_REQ, z=4, a=0.75)
ax.text(LEFT_X+1.0, lbl_y, '必需字段 ×5', fontsize=16, fontweight='bold',
        color='white', ha='center', va='center', fontproperties=CF, zorder=5)

# ===== 右列标题：可选字段 =====
rbox(ax, RIGHT_X-0.1, lbl_y-0.35, 2.2, 0.7, 0.08, C_OPT, z=4, a=0.75)
ax.text(RIGHT_X+1.0, lbl_y, '可选字段', fontsize=16, fontweight='bold',
        color='white', ha='center', va='center', fontproperties=CF, zorder=5)

# 根到两列的分叉线
root_bot = (ROOT_X+ROOT_W/2, ROOT_Y)
conn_branch(ax, ROOT_X+ROOT_W/2, ROOT_Y, [LEFT_X+1.1, RIGHT_X+1.1], [lbl_y+0.35, lbl_y+0.35])

# ===== 必需字段 =====
req_fields = [
    ('image_path', 'string', '图片相对路径，正斜杠分隔，如 "张三/img_001.jpg"'),
    ('bbox', 'object', '人脸边界框 {x, y, width, height}，x≥0, y≥0, w>0, h>0'),
    ('landmarks', 'object', '5个关键点 {left_eye, right_eye, nose_tip, left_mouth, right_mouth}，各为 [x,y]'),
    ('annotator', 'enum', '标注者类型："auto"（全自动）、"auto_partial"（仅bbox）、"manual"（人工修正）'),
    ('date', 'string', '标注日期，格式 YYYY-MM-DD，正则 ^\\d{4}-\\d{2}-\\d{2}$'),
]

start_y = lbl_y - 0.55
for i, (name, t, desc) in enumerate(req_fields):
    cy = start_y - i*(CARD_H + ROW_GAP)
    field_card(ax, LEFT_X, cy, CARD_W, CARD_H, name, t, desc, C_REQ, is_required=True)

# ===== bbox 子字段展开 =====
bbox_y = start_y - 1*(CARD_H + ROW_GAP)  # bbox所在卡片
bbox_card_bot = bbox_y
bbox_sub_x = LEFT_X + 0.5
bbox_sub_y = bbox_card_bot - 0.15
sub_field(ax, bbox_sub_x, bbox_sub_y, 'x', 'integer (≥0)', C_REQ)
sub_field(ax, bbox_sub_x+4.0, bbox_sub_y, 'y', 'integer (≥0)', C_REQ)
sub_field(ax, bbox_sub_x+8.0, bbox_sub_y, 'width', 'integer (>0)', C_REQ)
sub_field(ax, bbox_sub_x, bbox_sub_y-0.28, 'height', 'integer (>0)', C_REQ)
conn_line(ax, bbox_sub_x-0.2, bbox_card_bot, bbox_sub_x-0.2, bbox_sub_y+0.08, LT)
conn_line(ax, bbox_sub_x+3.8, bbox_card_bot, bbox_sub_x+3.8, bbox_sub_y+0.08, LT)

# ===== landmarks 子字段展开 =====
lm_y = start_y - 2*(CARD_H + ROW_GAP)
lm_card_bot = lm_y
lm_sub_y = lm_card_bot - 0.15
lm_sub_x = LEFT_X + 0.5
sub_field(ax, lm_sub_x, lm_sub_y, 'left_eye', '[x, y] 或 [-1,-1]', C_REQ)
sub_field(ax, lm_sub_x+5.5, lm_sub_y, 'right_eye', '[x, y]', C_REQ)
sub_field(ax, lm_sub_x, lm_sub_y-0.28, 'nose_tip', '[x, y]', C_REQ)
sub_field(ax, lm_sub_x+5.5, lm_sub_y-0.28, 'left_mouth', '[x, y]', C_REQ)
sub_field(ax, lm_sub_x, lm_sub_y-0.56, 'right_mouth', '[x, y]', C_REQ)
conn_line(ax, lm_sub_x-0.2, lm_card_bot, lm_sub_x-0.2, lm_sub_y+0.08, LT)

# ===== 可选字段 =====
opt_fields = [
    ('confidence', 'float (0~1)', '人脸检测置信度，仅在 auto / auto_partial 模式下出现'),
    ('quality', 'object', '质量评分详情：{quality_score, quality_label, blur_score, brightness_score, pose_score}'),
    ('reviewer', 'string', '审核者标识，记录执行人工审核的人员'),
    ('reviewed_from', 'string', '原始标注来源路径，追溯审核前的标注文件'),
    ('modifications', 'array', '修改记录列表，每项含 {field, old_value, new_value}，实现变更追溯'),
    ('extra', '—', '（可扩展：未来按需增加自定义字段，Schema 允许 additionalProperties）'),
]

for i, (name, t, desc) in enumerate(opt_fields):
    cy = start_y - i*(CARD_H + ROW_GAP)
    field_card(ax, RIGHT_X, cy, CARD_W, CARD_H, name, t, desc, C_OPT, is_required=False)

# ===== quality 子字段 =====
q_y = start_y - 1*(CARD_H + ROW_GAP)
q_sub_y = q_y - 0.15
q_sub_x = RIGHT_X + 0.5
sub_field(ax, q_sub_x, q_sub_y, 'quality_score', 'float (0~1)', C_OPT)
sub_field(ax, q_sub_x+5.5, q_sub_y, 'quality_label', 'good/acceptable/poor', C_OPT)
sub_field(ax, q_sub_x, q_sub_y-0.28, 'blur / brightness / pose_score', 'float (各维度0~1)', C_OPT)
conn_line(ax, q_sub_x-0.2, q_y, q_sub_x-0.2, q_sub_y+0.08, LT)

# ===== 图注 =====
ly = 0.5
for i,(lb,lc) in enumerate([('必需字段',C_REQ),('可选字段',C_OPT),('根节点',C_ROOT)]):
    lx = FW/2 - 2.5 + i*3.2
    rbox(ax, lx, ly, 0.45, 0.45, 0.06, lc, z=5, a=0.6)
    ax.text(lx+0.60, ly+0.22, lb, fontsize=13, color=TX,
            ha='left', va='center', fontproperties=CF, zorder=5)

# ===== 保存 =====
out = os.path.join(r'E:\face_annotation_project\output\charts', '图3-4_JSON字段结构示意图.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG, edgecolor='none', pad_inches=0.3)
plt.close(fig)
print(f'OK: {out}')
