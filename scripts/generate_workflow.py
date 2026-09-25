"""
整体工作流程图 v5 — 精确业务逻辑，廊道走线，不穿心
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
CA   = '#FFC107'; BG = '#FFFFFF'; TX = '#212121'; LT = '#888888'
C_PRE  = '#3F51B5'; C_MID = '#009688'; C_POST = '#00695C'
C_OPT_BG = '#FFFDE7'; C_OPT_BD = '#FFC107'

# ===== 画布 =====
FW, FH = 36, 22
DPI = 300
fig = plt.figure(figsize=(FW, FH), facecolor=BG)
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0,FW); ax.set_ylim(0,FH); ax.set_aspect('equal'); ax.axis('off')

ML, MR = 2.8, 2.2; CW = FW - ML - MR
SW = 4.5; SH = 2.0        # 主步骤
OW = 3.2; OH = 1.5         # 可选步骤（明显小于主步骤）

# ===== 标题 =====
ax.text(FW/2, FH-0.7, '系统整体工作流程图', fontsize=40, fontweight='bold',
        color=TX, ha='center', va='top', fontproperties=CF)
ax.text(FW/2, FH-1.3, '10-Step Pipeline with Optional Branches',
        fontsize=16, color=LT, ha='center', va='top', style='italic')
ax.plot([FW/2-3, FW/2+3], [FH-1.55, FH-1.55], color=CA, lw=2.5, solid_capstyle='round', zorder=5)

# ===== y 坐标（给足间隙） =====
R1Y = 16.0      # 行1 步骤1-4          [16.0, 18.0]
O1Y = 12.8      # 可选3.5/3.6          [12.8, 14.3]
R2Y = 9.2       # 行2 步骤5-6          [9.2,  11.2]
O2Y = 6.0       # 可选6.5              [6.0,  7.5]
R3Y = 3.5       # 行3 步骤7-10         [3.5,  5.5]

# 卡片边界
R1T, R1B = R1Y+SH, R1Y    # 18.0, 16.0
O1T, O1B = O1Y+OH, O1Y    # 14.3, 12.8
R2T, R2B = R2Y+SH, R2Y    # 11.2, 9.2
O2T, O2B = O2Y+OH, O2Y    # 7.5,  6.0
R3T, R3B = R3Y+SH, R3Y    # 5.5,  3.5

# ===== 右侧廊道 =====
CORRIDOR_X = ML + CW - 0.5  # ≈ 33.1，所有卡片最右侧之外的通路

# ===== 水平位置 =====
def even_x(n, w, cw, ml):
    g = (cw - n*w) / (n+1)
    return [ml + g + i*(w+g) for i in range(n)]

def left_x(n, w, cw, ml, start_offset):
    """左对齐排列"""
    return [ml + start_offset + i*(w+1.5) for i in range(n)]

r1 = even_x(4, SW, CW, ML)        # 步骤1-4 均匀分布
r2 = left_x(2, SW, CW, ML, 2.0)   # 步骤5-6 靠左排列
r3 = even_x(4, SW, CW, ML)        # 步骤7-10 均匀分布

# ===== 辅助函数 =====
def stage_tag(ax, y_t, y_b, color, name):
    lw, lh = 2.0, y_t - y_b
    ax.add_patch(FancyBboxPatch((0.4, y_b), lw, lh,
        boxstyle=f"round,pad=0.02,rounding_size=0.14", mutation_scale=lw,
        facecolor=color, edgecolor='none', linewidth=0, zorder=8, alpha=0.85))
    ax.text(0.4+lw/2, y_b+lh/2, name, fontsize=20, fontweight='bold',
            color='white', ha='center', va='center', fontproperties=CF, zorder=9)

def card(ax, x, y, w, h, num, name, mod, color, optional=False):
    if optional:
        ax.add_patch(FancyBboxPatch((x,y), w, h,
            boxstyle=f"round,pad=0.01,rounding_size=0.08", mutation_scale=w,
            facecolor=C_OPT_BG, edgecolor=C_OPT_BD, linewidth=2.5, linestyle='--', zorder=4))
    else:
        ax.add_patch(FancyBboxPatch((x,y), w, h,
            boxstyle=f"round,pad=0.01,rounding_size=0.08", mutation_scale=w,
            facecolor=color, edgecolor='none', linewidth=0, zorder=4, alpha=0.15))
    ax.text(x+w/2, y+h-0.48, f'步骤{num}', fontsize=17, fontweight='bold',
            color=color if not optional else C_OPT_BD, ha='center', va='center',
            fontproperties=CF, zorder=5)
    ax.text(x+w/2, y+h-1.20, name, fontsize=22, fontweight='bold',
            color=TX, ha='center', va='center', fontproperties=CF, zorder=5)
    ax.text(x+w/2, y+0.52, mod, fontsize=14, color=LT,
            ha='center', va='center', fontproperties=CF, zorder=5)

def h_arrow(ax, x0, x1, y, color=CA):
    ax.annotate('', xy=(x1-0.06, y), xytext=(x0+0.06, y),
                arrowprops=dict(arrowstyle='->,head_width=0.35,head_length=0.25',
                color=color, lw=3.5), zorder=6)

def v_dash_arrow(ax, x, y0, y1):
    """向下虚线短箭头"""
    ax.annotate('', xy=(x, y1+0.04), xytext=(x, y0-0.04),
                arrowprops=dict(arrowstyle='->,head_width=0.25,head_length=0.18',
                color=C_OPT_BD, lw=2.2, ls='--'), zorder=6)

def v_dash_arrow_up(ax, x, y0, y1):
    """向上虚线短箭头"""
    ax.annotate('', xy=(x, y1-0.04), xytext=(x, y0+0.04),
                arrowprops=dict(arrowstyle='->,head_width=0.25,head_length=0.18',
                color=C_OPT_BD, lw=2.2, ls='--'), zorder=6)

def corner_arrow(ax, x0, y0, x1, y1, mid_y, color=CA, ls='-', lw=3.0):
    """L形拐角箭头：从(x0,y0)→向下到mid_y→水平到x1→垂直到达(x1,y1)"""
    verts = [(x0,y0),(x0,mid_y),(x1,mid_y),(x1,y1)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    ax.add_patch(FancyArrowPatch(
        path=Path(verts, codes),
        arrowstyle=f'->,head_width=0.40,head_length=0.28',
        color=color, lw=lw, ls=ls, zorder=6))


# ===== 阶段标签 =====
stage_tag(ax, R1T, O1B, C_PRE, '前期\n处理')
stage_tag(ax, R2T, O2B, C_MID, '中期\n校验')
stage_tag(ax, R3T, R3B, C_POST, '后期\n交付')


# ============================================================
# 第1阶段：前期处理 — 步骤1→2→3→4
# ============================================================
pre = [('1','数据采集','上传/拍照/URL/导入'),
       ('2','人脸检测','FaceDetector'),
       ('3','关键点提取','LandmarkExtractor'),
       ('4','标注生成','AnnotationGenerator')]
for i,(n,name,mod) in enumerate(pre):
    card(ax, r1[i], R1Y, SW, SH, n, name, mod, C_PRE)
    if i < 3:
        h_arrow(ax, r1[i]+SW, r1[i+1], R1Y+SH/2)

# ---- 步骤3.5 在步骤3正下方 ----
sc3 = r1[2] + SW/2
ox35 = sc3 - OW/2
card(ax, ox35, O1Y, OW, OH, '3.5', '质量评估', 'QualityFilter', C_PRE, optional=True)
v_dash_arrow(ax, sc3, R1B, O1T)        # 步骤3↓→3.5

# ---- 步骤3.6 在步骤4正下方 ----
sc4 = r1[3] + SW/2
ox36 = sc4 - OW/2
card(ax, ox36, O1Y, OW, OH, '3.6', '图片预处理', 'ImagePreprocessor', C_PRE, optional=True)
v_dash_arrow(ax, sc4, R1B, O1T)        # 步骤4↓→3.6


# ============================================================
# 拐角箭头：步骤4 → 步骤5
# 步骤4右下→向下到间隙→水平左移→从步骤5左侧进入
# ============================================================
sc5 = r2[0] + SW/2
# 水平段选在 O1B(12.8) 和 R2T(11.2) 之间的正中间
mid_45 = (O1B + R2T) / 2  # = 12.0
# 从步骤4右侧内部出发
vx_45 = r1[3] + SW - 0.3   # 步骤4右下角内侧
# 到达步骤5左侧（步骤5的x左边界）
tx_45 = r2[0] + 0.2         # 步骤5左边缘内侧
corner_arrow(ax, vx_45, R1B, tx_45, R2T, mid_45, CA)


# ============================================================
# 第2阶段：中期校验 — 步骤5→6
# ============================================================
mid = [('5','结构校验','Validator'),
       ('6','身份识别','FaceSignature')]
for i,(n,name,mod) in enumerate(mid):
    card(ax, r2[i], R2Y, SW, SH, n, name, mod, C_MID)
    if i < 1:
        h_arrow(ax, r2[i]+SW, r2[i+1], R2Y+SH/2)


# ============================================================
# 拐角箭头：步骤6 → 步骤7（主线，橙色实线）
# 步骤6右下→向下到间隙→水平左移→从步骤7左侧进入
# ============================================================
sc6 = r2[1] + SW/2  # 步骤6中心
sc7 = r3[0] + SW/2  # 步骤7中心
mid_67 = (O2B + R3T) / 2  # = (6.0+5.5)/2 = 5.75
vx_67 = r2[1] + SW - 0.3  # 步骤6右下角内侧
tx_67 = r3[0] + 0.2        # 步骤7左边缘内侧
corner_arrow(ax, vx_67, R2B, tx_67, R3T, mid_67, CA)


# ============================================================
# 可选步骤 6.5 — 步骤6正下方
# 虚线向下：步骤6→6.5  实线拐角：6.5→步骤7
# ============================================================
ox65 = sc6 - OW/2
card(ax, ox65, O2Y, OW, OH, '6.5', '标注纠错', 'AnnotationCorrector', C_MID, optional=True)
v_dash_arrow(ax, sc6, R2B, O2T)    # 步骤6↓→6.5（虚线）

# 6.5→7：从6.5右下角向下到间隙（比主线稍高0.2），水平左移，汇入步骤7
vx_657 = ox65 + OW - 0.3           # 步骤6.5右下角内侧
mid_657 = mid_67 + 0.25            # 比主线稍高，避免重叠
tx_657 = r3[0] + 0.8               # 步骤7左边缘内0.8英寸（汇入点略不同于主线）
corner_arrow(ax, vx_657, O2B, tx_657, R3T, mid_657, CA)


# ============================================================
# 第3阶段：后期交付 — 步骤7→8→9→10
# ============================================================
post = [('7','人工审核','ManualReviewer'),
        ('8','质量报告','QualityReport'),
        ('9','格式导出','FormatConverter'),
        ('10','可视化交付','Visualizer')]
for i,(n,name,mod) in enumerate(post):
    card(ax, r3[i], R3Y, SW, SH, n, name, mod, C_POST)
    if i < 3:
        h_arrow(ax, r3[i]+SW, r3[i+1], R3Y+SH/2)

# ===== 图注 =====
ly = 1.6
ax.add_patch(FancyBboxPatch((ML+15, ly-0.08), 0.55, 0.55,
    boxstyle="round,pad=0.01,rounding_size=0.08", mutation_scale=0.55,
    facecolor=C_OPT_BG, edgecolor=C_OPT_BD, linewidth=2.5, linestyle='--', zorder=5))
ax.text(ML+15.75, ly+0.20, '虚线框 = 可选步骤', fontsize=14, color=TX,
        ha='left', va='center', fontproperties=CF, zorder=5)
for i,(lb,lc) in enumerate([('前期处理',C_PRE),('中期校验',C_MID),('后期交付',C_POST)]):
    lx = ML+0.5+i*4.5
    ax.add_patch(FancyBboxPatch((lx, ly-0.08), 0.55, 0.55,
        boxstyle="round,pad=0.01,rounding_size=0.08", mutation_scale=0.55,
        facecolor=lc, edgecolor='none', linewidth=0, zorder=5, alpha=0.50))
    ax.text(lx+0.75, ly+0.20, lb, fontsize=14, color=TX,
            ha='left', va='center', fontproperties=CF, zorder=5)

# ===== 保存 =====
out = os.path.join(r'E:\face_annotation_project\output\charts', '图3-6_整体工作流程图.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG, edgecolor='none', pad_inches=0.3)
plt.close(fig)
print(f'OK: {out}')
print(f'    间隙: R1-O1={R1B-O1T:.1f} O1-R2={O1B-R2T:.1f} R2-O2={R2B-O2T:.1f} O2-R3={O2B-R3T:.1f}')
print(f'    拐角水平段: 4→5 y={mid_45:.1f}  6→7 y={mid_67:.1f}  6.5→7 y={mid_657:.1f}')
print(f'    步骤5 x=({r2[0]:.1f}~{r2[0]+SW:.1f}) 步骤6 x=({r2[1]:.1f}~{r2[1]+SW:.1f})')
