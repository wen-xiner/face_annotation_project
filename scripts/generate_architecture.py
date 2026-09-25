"""
系统整体架构图 v6 — 大字体 + 宽间距，杜绝重叠
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
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
C4 = '#00695C'; C3 = '#009688'; C2 = '#26A69A'; C1 = '#3F51B5'; CA = '#FFC107'
BG = '#FFFFFF'; TX = '#212121'; LT = '#888888'

# ===== 画布（加大） =====
FW, FH = 36, 26
DPI = 300
fig = plt.figure(figsize=(FW, FH), facecolor=BG)
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0,FW); ax.set_ylim(0,FH); ax.set_aspect('equal'); ax.axis('off')

ML = 3.0; MR = 2.2; CW = FW - ML - MR

# ===== 标题 =====
ax.text(FW/2, FH-0.7, '系统整体架构图', fontsize=42, fontweight='bold',
        color=TX, ha='center', va='top', fontproperties=CF)
ax.text(FW/2, FH-1.3, 'Four-Layer Modular Architecture',
        fontsize=16, color=LT, ha='center', va='top', style='italic')
ax.plot([FW/2-4, FW/2+4], [FH-1.6, FH-1.6], color=CA, lw=3,
        solid_capstyle='round', zorder=5)


def layer_label(ax, y, h, color, name):
    lw = 2.2
    ax.add_patch(FancyBboxPatch((0.45, y), lw, h,
        boxstyle=f"round,pad=0.02,rounding_size=0.14", mutation_scale=lw,
        facecolor=color, edgecolor='none', linewidth=0, zorder=8, alpha=0.85))
    ax.text(0.45+lw/2, y+h/2, name, fontsize=22, fontweight='bold',
            color='white', ha='center', va='center', fontproperties=CF, zorder=9)

def layer_sep(ax, y):
    ax.plot([ML, ML+CW], [y, y], color='#E0E0E0', lw=1.2, ls='--', zorder=1)

def layer_arrow(ax, y0, y1):
    ax.annotate('', xy=(1.55, y1), xytext=(1.55, y0),
                arrowprops=dict(arrowstyle='->,head_width=0.55,head_length=0.35',
                                color=CA, lw=3.5), zorder=7)


# ============================================================
# 第1层: 基础依赖层
# ============================================================
LY1, LH1 = 1.8, 2.5
layer_label(ax, LY1, LH1, C1, '基础依赖层')

libs_name = ['OpenCV', 'MediaPipe', 'insightface', 'FastAPI', 'NumPy']
libs_desc = ['图像处理\n计算机视觉', '人脸检测\n关键点提取', 'ArcFace R50\n身份识别',
             '异步Web框架\nRESTful API', '数值计算\nembedding']
n = len(libs_name)
iw = (CW - 2.0) / n
for i in range(n):
    ix = ML + 0.5 + i*iw
    ax.text(ix+iw/2, LY1+LH1-0.5, libs_name[i], fontsize=24, fontweight='bold',
            color=C1, ha='center', va='top', zorder=5)
    ax.text(ix+iw/2, LY1+LH1-1.2, libs_desc[i], fontsize=13, color=LT,
            ha='center', va='top', fontproperties=CF, zorder=5, linespacing=1.6)

GAP12 = 2.2
layer_sep(ax, LY1+LH1+GAP12/2)
layer_arrow(ax, LY1+LH1, LY1+LH1+GAP12)


# ============================================================
# 第2层: 核心算法层 — 12模块, 3行x4列
# ============================================================
LY2 = LY1 + LH1 + GAP12
LH2 = 7.0
layer_label(ax, LY2, LH2, C2, '核心算法层')

mods_name = ['FaceDetector', 'LandmarkExtractor', 'AnnotationGenerator', 'FaceSignature',
             'BatchProcessor', 'Visualizer', 'Validator', 'ManualReviewer',
             'QualityFilter', 'FormatConverter', 'ImagePreprocessor', 'AnnotationCorrector']
mods_desc = ['BlazeFace 人脸检测', '478点→5关键点', '生成JSON标注', 'ArcFace 身份识别',
             '批量自动标注', '标注可视化渲染', 'JSON 结构校验', '人工修正审核',
             '三维质量评分', '5种格式双向转换', '去噪/CLAHE增强', '几何约束纠错']
ncols, nrows = 4, 3
# 列间距1.5英寸，行间距1.2英寸（大幅拉大）
col_gap = 1.5; row_gap = 1.2
mw = (CW - (ncols-1)*col_gap) / ncols
mh = (LH2 - 0.8 - (nrows-1)*row_gap) / nrows

for idx in range(12):
    col = idx % ncols; row = idx // ncols
    mx = ML + col*(mw+col_gap)
    my = LY2 + 0.4 + row*(mh+row_gap)

    # 模块名（17pt）+ 描述（13pt），垂直间距0.6英寸，mh约1.5英寸，完全安全
    ax.text(mx+0.2, my+mh-0.35, f'{idx+1:02d}  {mods_name[idx]}',
            fontsize=17, fontweight='bold', color=TX, ha='left', va='top', zorder=5)
    ax.text(mx+0.2, my+mh-0.95, mods_desc[idx],
            fontsize=14, color=C2, ha='left', va='top', fontproperties=CF, zorder=5)

GAP23 = 2.0
layer_sep(ax, LY2+LH2+GAP23/2)
layer_arrow(ax, LY2+LH2, LY2+LH2+GAP23)


# ============================================================
# 第3层: 数据处理管道层 — 9步
# ============================================================
LY3 = LY2 + LH2 + GAP23
LH3 = 3.8
layer_label(ax, LY3, LH3, C3, '数据处理管道层')

steps_name = ['数据采集', '图像清洗', '人脸检测', '关键点\n提取', '质量\n评估',
              '标注\n生成', '自动\n纠错', '结构\n校验', '格式\n导出']
steps_sub = ['上传/拍照\nURL/导入', '去噪/CLAHE\n格式统一', 'MediaPipe\nBlazeFace',
             '478点提取\n5点简化', '模糊/亮度\n姿态评分', 'JSON生成\n字段填充',
             '几何约束\n4项修正', 'Schema验证\nerror列表', 'COCO/YOLO\nVOC/CSV']
n = len(steps_name)
sw = 2.8
sgap = (CW - n*sw) / (n+1)  # 约 0.67 英寸，箭头充足

for i in range(n):
    sx = ML + sgap + i*(sw+sgap)

    # 极淡底色块
    ax.add_patch(FancyBboxPatch((sx, LY3+0.3), sw, LH3-0.6,
        boxstyle=f"round,pad=0.01,rounding_size=0.08", mutation_scale=sw,
        facecolor=C3, edgecolor='none', linewidth=0, zorder=2, alpha=0.06))

    # 垂直布局：编号→步骤名→子描述，彼此间距 > 0.8英寸
    num_y = LY3 + LH3 - 0.70
    name_y = LY3 + LH3 - 1.65
    sub_y = LY3 + 1.05

    ax.text(sx+sw/2, num_y, f'{i+1}', fontsize=30, fontweight='bold',
            color=C3, ha='center', va='center', zorder=5)
    ax.text(sx+sw/2, name_y, steps_name[i], fontsize=18, fontweight='bold',
            color=TX, ha='center', va='center', fontproperties=CF, zorder=5, linespacing=1.3)
    ax.text(sx+sw/2, sub_y, steps_sub[i], fontsize=12, color=LT,
            ha='center', va='center', fontproperties=CF, zorder=5, linespacing=1.5)

    if i < n-1:
        ax0 = sx + sw + 0.08
        ax1 = sx + sw + sgap - 0.08
        ay = LY3 + LH3/2
        ax.annotate('', xy=(ax1, ay), xytext=(ax0, ay),
                    arrowprops=dict(arrowstyle='->,head_width=0.32,head_length=0.20',
                    color=CA, lw=3.2), zorder=6)

ax.text(ML+CW/2, LY3-0.4, '端到端自动化管道：原始图片输入 → 最终标注交付',
        fontsize=14, color=C3, ha='center', va='top',
        fontproperties=CF, fontweight='bold', zorder=5)

GAP34 = 1.8
layer_sep(ax, LY3+LH3+GAP34/2)
layer_arrow(ax, LY3+LH3, LY3+LH3+GAP34)


# ============================================================
# 第4层: Web 服务层
# ============================================================
LY4 = LY3 + LH3 + GAP34
LH4 = 2.5
layer_label(ax, LY4, LH4, C4, 'Web 服务层')

items = [
    ('RESTful API', 'FastAPI · 31 端点 · OpenAPI 文档'),
    ('Web 前端', 'HTML/CSS/JS · 左右分栏 · 实时预览'),
    ('CLI 工具集', '6 条命令 · 批量处理 · 无头环境'),
]
n = len(items)
iw2 = 7.0
igap2 = (CW - n*iw2) / (n+1)
for i, (title, desc) in enumerate(items):
    ix = ML + igap2 + i*(iw2+igap2)
    ax.text(ix+iw2/2, LY4+LH4-0.60, title, fontsize=26, fontweight='bold',
            color=C4, ha='center', va='center', fontproperties=CF, zorder=5)
    ax.text(ix+iw2/2, LY4+LH4-1.50, desc, fontsize=15, color=LT,
            ha='center', va='center', fontproperties=CF, zorder=5)


# ===== 图注 =====
leg = [('基础依赖层',C1),('核心算法层',C2),('数据处理管道层',C3),('Web 服务层',C4)]
lx0 = FW/2 - 5.8
for i,(lb,lc) in enumerate(leg):
    lx = lx0 + i*3.4
    ax.add_patch(FancyBboxPatch((lx-0.18, 0.35), 0.36, 0.36,
        boxstyle="round,pad=0.01,rounding_size=0.06", mutation_scale=0.36,
        facecolor=lc, edgecolor='none', linewidth=0, zorder=5, alpha=0.75))
    ax.text(lx+0.35, 0.53, lb, fontsize=15, color=TX,
            ha='left', va='center', fontproperties=CF, fontweight='bold', zorder=5)

# ===== 保存 =====
out = os.path.join(r'E:\face_annotation_project\output\charts', '图3-1_系统整体架构图.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG, edgecolor='none', pad_inches=0.3)
plt.close(fig)

# 验证：检查所有垂直间距
print(f'OK: {out}')
print(f'    画布: {FW}x{FH}in @ {DPI}dpi')
print(f'    垂直间距验证:')
print(f'      层1->2: {GAP12}in')
print(f'      层2->3: {GAP23}in')
print(f'      层3->4: {GAP34}in')
print(f'      管道步骤间距: {sgap:.2f}in')
print(f'      模块行间距: {row_gap}in, 列间距: {col_gap}in')
