"""
数据采集流程图 — 四种渠道汇聚到统一数据集目录
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
C1 = '#3F51B5'; C2 = '#009688'; C3 = '#26A69A'; C4 = '#00695C'
CA  = '#FFC107'; BG = '#FFFFFF'; TX = '#212121'; LT = '#888888'

# ===== 画布 =====
FW, FH = 32, 16
DPI = 300
fig = plt.figure(figsize=(FW, FH), facecolor=BG)
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0,FW); ax.set_ylim(0,FH); ax.set_aspect('equal'); ax.axis('off')

# ===== 标题 =====
ax.text(FW/2, FH-0.7, '数据采集流程图', fontsize=38, fontweight='bold',
        color=TX, ha='center', va='top', fontproperties=CF)
ax.text(FW/2, FH-1.2, 'Multi-Source Data Collection → Unified Dataset Directory',
        fontsize=15, color=LT, ha='center', va='top', style='italic')
ax.plot([FW/2-3, FW/2+3], [FH-1.45, FH-1.45], color=CA, lw=2.5, solid_capstyle='round', zorder=5)

# ===== 辅助 =====
def rbox(ax, x, y, w, h, r, fc, ec='none', lw=0, zorder=1, a=1):
    ax.add_patch(FancyBboxPatch((x,y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={r}", mutation_scale=max(w,1),
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder, alpha=a))

def arrow_to_center(ax, x0, y0, x1, y1, color=CA):
    """从渠道指向中心"""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->,head_width=0.45,head_length=0.30',
                color=color, lw=3.5), zorder=4)

# ===== 布局 =====
# 左侧4个渠道卡片垂直排列，右侧中心汇聚目标
CH_W, CH_H = 12.0, 2.0    # 渠道卡片
CENTER_W, CENTER_H = 7.0, 3.5  # 中心汇聚卡片

LEFT_X = 2.5
RIGHT_X = LEFT_X + CH_W + 1.5
CH_START_Y = 4.5
CH_GAP = 0.8

CENT_X = RIGHT_X + 1.5
CENT_Y = FH/2 - CENTER_H/2 + 0.5

# ===== 4个渠道卡片 =====
channels = [
    ('本地文件上传', 'Web界面拖拽/点击  →  UUID重命名  →  web_app/uploads/',
     '支持 jpg/jpeg/png/bmp/tiff/webp 七种格式，前端校验扩展名白名单', C1),
    ('摄像头拍照录像', 'getUserMedia API  →  1280×720  →  JPEG(0.92) 压缩',
     '录像模式按可配置时间间隔（默认3秒）自动连续截图', C2),
    ('URL 远程下载', 'urllib + ThreadPoolExecutor(5线程)  →  30秒超时',
     '设置 User-Agent 标识，下载后校验 ≥1KB 及格式合法性', C3),
    ('外部文件夹批量导入', '--from-dir 参数  →  copy/symlink  →  --organize 归类',
     '保留源目录子文件夹结构自动识别为人物姓名', C4),
]

for i, (name, flow, desc, color) in enumerate(channels):
    cy = CH_START_Y + i*(CH_H + CH_GAP)

    # 卡片背景
    rbox(ax, LEFT_X, cy, CH_W, CH_H, 0.12, color, zorder=2, a=0.08)

    # 左侧色条
    rbox(ax, LEFT_X, cy, 0.10, CH_H, 0.05, color, zorder=3, a=0.8)

    # 序号圆圈
    from matplotlib.patches import Circle
    ax.add_patch(Circle((LEFT_X+0.45, cy+CH_H-0.45), 0.22, fc=color, ec='none', zorder=4))
    ax.text(LEFT_X+0.45, cy+CH_H-0.45, str(i+1), fontsize=13, fontweight='bold',
            color='white', ha='center', va='center', zorder=5)

    # 渠道名称
    ax.text(LEFT_X+0.80, cy+CH_H-0.45, name, fontsize=18, fontweight='bold',
            color=color, ha='left', va='center', fontproperties=CF, zorder=4)

    # 技术流程
    ax.text(LEFT_X+0.30, cy+CH_H-1.05, flow, fontsize=11.5, color=TX,
            ha='left', va='center', fontproperties=CF, zorder=4)

    # 补充说明
    ax.text(LEFT_X+0.30, cy+0.40, desc, fontsize=10, color=LT,
            ha='left', va='center', fontproperties=CF, zorder=4)

    # 汇聚箭头：从卡片右侧中心 → 中心卡片左侧
    arrow_y_out = cy + CH_H/2
    arrow_x_in = CENT_X + 0.15
    # 目标y略分散到中心卡片的不同位置
    target_offsets = [-1.0, -0.3, 0.3, 1.0]
    arrow_y_in = CENT_Y + CENTER_H/2 + target_offsets[i]
    arrow_to_center(ax, LEFT_X+CH_W, arrow_y_out, arrow_x_in, arrow_y_in)

# ===== 中心汇聚卡片 =====
rbox(ax, CENT_X, CENT_Y, CENTER_W, CENTER_H, 0.18, '#E8EAF6', ec='#3F51B5', lw=2.5, zorder=5, a=0.5)
rbox(ax, CENT_X, CENT_Y+CENTER_H-0.7, CENTER_W, 0.7, 0.18, '#3F51B5', zorder=6, a=0.85)
# 覆盖底部圆角
ax.add_patch(plt.Rectangle((CENT_X+0.02, CENT_Y+CENTER_H-0.7), CENTER_W-0.04, 0.22,
              fc='#E8EAF6', ec='none', zorder=7, alpha=0.5))

ax.text(CENT_X+CENTER_W/2, CENT_Y+CENTER_H-0.35, '统一数据集目录',
        fontsize=20, fontweight='bold', color='white',
        ha='center', va='center', fontproperties=CF, zorder=8)

# 内部说明
ax.text(CENT_X+CENTER_W/2, CENT_Y+CENTER_H-1.50, 'dataset/raw_images/',
        fontsize=15, color='#3F51B5', ha='center', va='center',
        fontweight='bold', zorder=6)

ax.text(CENT_X+CENTER_W/2, CENT_Y+CENTER_H-2.25, '按人物姓名分文件夹组织\n5位人物 · 84+张原始图片',
        fontsize=12, color=TX, ha='center', va='center',
        fontproperties=CF, zorder=6, linespacing=1.6)

ax.text(CENT_X+CENTER_W/2, CENT_Y+0.55, 'web_app/uploads/ (UUID命名)',
        fontsize=11, color=LT, ha='center', va='center',
        style='italic', zorder=6)

# ===== 中心右侧输出箭头 → 下一阶段 =====
out_x = CENT_X + CENTER_W
out_y = CENT_Y + CENTER_H/2
ax.annotate('', xy=(out_x+1.6, out_y), xytext=(out_x+0.1, out_y),
            arrowprops=dict(arrowstyle='->,head_width=0.55,head_length=0.35',
            color=CA, lw=4.0), zorder=5)
ax.text(out_x+0.85, out_y-0.7, '进入\n标注流程', fontsize=14, color=CA,
        ha='center', va='top', fontproperties=CF, fontweight='bold', zorder=5)

# ===== 图注 =====
ly = 1.4
for i,(lb,lc) in enumerate([('本地文件上传',C1),('摄像头拍照录像',C2),
                              ('URL远程下载',C3),('外部文件夹导入',C4)]):
    lx = FW/2 - 5.5 + i*3.2
    rbox(ax, lx, ly, 0.40, 0.40, 0.06, lc, zorder=5, a=0.7)
    ax.text(lx+0.55, ly+0.20, lb, fontsize=12, color=TX,
            ha='left', va='center', fontproperties=CF, zorder=5)

# ===== 保存 =====
out = os.path.join(r'E:\face_annotation_project\output\charts', '图4-1_数据采集流程图.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG, edgecolor='none', pad_inches=0.3)
plt.close(fig)
print(f'OK: {out}')
