"""
生成报告所需的三张 matplotlib 图表：
  1. 性能测试柱状图（5.5 节）
  2. 标注准确率可视化图表（5.6 节）
  3. 项目甘特图（7.5 节）
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np
import json
import os

# ── 全局样式设置 ──────────────────────────────────────────────
plt.rcParams["font.family"]     = "Microsoft YaHei"
plt.rcParams["font.size"]       = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"]     = 200
plt.rcParams["savefig.bbox"]    = "tight"
plt.rcParams["savefig.facecolor"] = "white"

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "output", "charts")
os.makedirs(OUTDIR, exist_ok=True)

# ── 加载实际质量报告数据 ──────────────────────────────────────
report_path = os.path.join(os.path.dirname(OUTDIR), "quality_report.json")
quality_data = None
if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        quality_data = json.load(f)


# ===================================================================
#  图1：性能测试柱状图
# ===================================================================
def fig1_performance_bar():
    """横向柱状图，展示单张图片各处理环节耗时。"""
    # 数据来自报告表5-5（单张平均耗时，单位：秒）
    steps = [
        "人脸检测+关键点提取",
        "图片清洗\n(双边滤波+CLAHE)",
        "质量评估\n(三维度评估)",
        "标注自动纠错",
        "身份识别\n(Embedding提取)",
        "JSON导出",
    ]
    times = [0.3, 0.1, 0.05, 0.008, 0.2, 0.005]

    # 颜色分组：模型推理 / 图像处理 / 数值+I/O
    colors = ["#E74C3C", "#3498DB", "#3498DB", "#2ECC71", "#E74C3C", "#95A5A6"]

    y = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.barh(y, times, height=0.6, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(steps)
    ax.invert_yaxis()
    ax.set_xlabel("单张平均耗时（秒）", fontsize=12)
    ax.set_title("各处理环节单张图片耗时对比", fontsize=15, fontweight="bold", pad=15)

    # 数值标签
    for bar, t in zip(bars, times):
        label = f"{t:.3f}s" if t < 0.01 else f"{t:.2f}s"
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10, fontweight="bold")

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E74C3C", label="模型推理（MediaPipe / insightface）"),
        Patch(facecolor="#3498DB", label="图像处理（OpenCV）"),
        Patch(facecolor="#2ECC71", label="纯数值计算"),
        Patch(facecolor="#95A5A6", label="文件 I/O"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#ccc")

    ax.set_xlim(0, max(times) * 1.35)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()

    path = os.path.join(OUTDIR, "图5-7_各处理环节性能对比.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] fig1 saved -> {path}")


# ===================================================================
#  图2：标注准确率与质量分布
# ===================================================================
def fig2_accuracy_quality():
    """左侧饼图（质量等级分布）+ 右侧柱状图（各维度平均分）"""
    if quality_data is None:
        print("[!] quality_report.json 未找到，跳过图2")
        return

    s = quality_data["summary"]
    qd = s["quality_distribution"]
    sub = s["sub_scores"]

    fig = plt.figure(figsize=(15, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.55], wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])  # 饼图
    ax2 = fig.add_subplot(gs[0, 1])  # 柱状图
    ax3 = fig.add_subplot(gs[0, 2])  # 统计表（独立，不遮挡）

    # ── 左：质量等级饼图 ──
    labels = [f"good\n(≥0.7)", f"acceptable\n(0.4–0.7)"]
    sizes  = [qd["by_label"]["good"], qd["by_label"]["acceptable"]]
    colors_pie = ["#27AE60", "#F39C12"]
    explode = (0.02, 0.02)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct="%1.1f%%", startangle=140, pctdistance=0.55,
        textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(13)
    # 中心加总数
    ax1.text(0, -0.05, f"共 {s['validation']['total']} 张", ha="center", va="center",
             fontsize=13, fontweight="bold", color="#2C3E50")
    ax1.set_title("质量等级分布", fontsize=14, fontweight="bold", pad=18)

    # ── 右：各维度平均分柱状图 ──
    dims = ["模糊度", "亮度", "姿态", "人脸占比", "关键点位置"]
    scores = [sub["mean_blur"], sub["mean_brightness"], sub["mean_pose"],
              sub["mean_face_ratio"], sub["mean_landmark_position"]]
    bar_colors = ["#3498DB", "#3498DB", "#3498DB", "#9B59B6", "#2ECC71"]

    x = np.arange(len(dims))
    bars = ax2.bar(x, scores, width=0.55, color=bar_colors, edgecolor="white", linewidth=0.6)

    # 综合均分横线
    mean_all = qd["mean_score"]
    ax2.axhline(y=mean_all, color="#E74C3C", linewidth=2, linestyle="--", alpha=0.8)
    ax2.text(len(dims) - 0.45, mean_all + 0.012, f"综合均分 {mean_all:.3f}",
             color="#E74C3C", fontsize=10, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(dims, fontsize=11)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("平均评分（0–1）", fontsize=12)
    ax2.set_title("各维度平均评分", fontsize=14, fontweight="bold", pad=15)

    for bar, sc in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                 f"{sc:.3f}", ha="center", fontsize=10, fontweight="bold")

    # ── 统计汇总：放在第三栏独立表格 ──
    ax3.axis("off")
    table_data = [
        ["有效文件", f"{s['validation']['valid']} 张"],
        ["无效文件", f"{s['validation']['invalid']} 张"],
        ["最高分",   f"{qd['max_score']:.3f}"],
        ["最低分",   f"{qd['min_score']:.3f}"],
        ["accept",  f"{qd['by_verdict']['accept']} 张"],
        ["review",   f"{qd['by_verdict']['review']} 张"],
    ]
    col_labels = ["指标", "数值"]
    tbl = ax3.table(cellText=table_data, colLabels=col_labels,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.6)
    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_facecolor("#D5E8D4")
            cell.get_text().set_fontweight("bold")
        elif key[1] == 0:
            cell.set_facecolor("#F8F9FA")
            cell.get_text().set_fontweight("bold")
    ax3.set_title("统计汇总", fontsize=12, fontweight="bold", pad=10)

    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    path = os.path.join(OUTDIR, "图5-8_标注准确率与质量分布.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] 图2 已保存 → {path}")


# ===================================================================
#  图3：项目甘特图
# ===================================================================
def fig3_gantt():
    """用 broken_barh 绘制项目里程碑甘特图。"""
    # 阶段名、开始天数、持续天数、颜色（总跨度约12个工作日，即两周）
    phases = [
        ("需求分析与技术选型",     0,   2,  "#2C3E50"),
        ("v1.0 核心标注引擎开发",  1.5, 3,  "#2980B9"),
        ("v2.0 Web界面与审核开发", 4,   3,  "#3498DB"),
        ("v3.0 质量评估与格式转换", 6,   2.5,"#27AE60"),
        ("数据采集与自动标注",     7.5, 2,  "#F39C12"),
        ("质量检验与人工审核",     9,   1.5,"#E67E22"),
        ("报告撰写与修改",         10,  2.5,"#8E44AD"),
    ]

    fig, ax = plt.subplots(figsize=(14, 5.0))

    y_positions = []
    for i, (name, start, duration, color) in enumerate(phases):
        y = i
        ax.broken_barh([(start, duration)], (y - 0.35, 0.7),
                       facecolors=color, edgecolor="white", linewidth=0.5, alpha=0.92)
        # 阶段名放在条形右侧
        ax.text(start + duration + 0.15, y, name, va="center", fontsize=10.5,
                fontweight="bold", color="#2C3E50")
        y_positions.append(y)

    # 里程碑标记（菱形）
    milestones = [
        (3,   "v1.0", "#2980B9"),
        (5.5, "v2.0", "#3498DB"),
        (8,   "v3.0", "#27AE60"),
        (12,  "终稿", "#8E44AD"),
    ]
    for mx, mlabel, mcolor in milestones:
        ax.plot(mx, -0.55, marker="D", color=mcolor, markersize=8, clip_on=False)
        ax.text(mx, -0.95, mlabel, ha="center", fontsize=8.5, color=mcolor,
                fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * len(y_positions))
    ax.set_xlabel("工作日", fontsize=12)
    ax.set_title("项目里程碑甘特图（总工期约两周）", fontsize=15, fontweight="bold", pad=15)

    # 顶部加日期参考轴
    ax2_top = ax.twiny()
    ax2_top.set_xlim(ax.get_xlim())
    ax2_top.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax2_top.set_xticklabels(["第1天", "第3天", "第5天", "第7天", "第9天", "第11天", "第13天"],
                             fontsize=9, color="#7F8C8D")
    ax2_top.spines["top"].set_visible(False)

    ax.set_ylim(-1.2, len(phases) - 0.3)
    ax.set_xlim(-0.3, 13.8)
    ax.set_xticks(range(0, 14, 2))
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()

    path = os.path.join(OUTDIR, "图7-1_项目里程碑甘特图.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] 图3 已保存 → {path}")


# ===================================================================
#  Main
# ===================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  报告图表生成器")
    print("=" * 55)
    fig1_performance_bar()
    fig2_accuracy_quality()
    fig3_gantt()
    print(f"\n所有图表已输出至: {OUTDIR}")
