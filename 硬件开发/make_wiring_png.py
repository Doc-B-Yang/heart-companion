# -*- coding: utf-8 -*-
"""
把四模态接线关系渲染成一张高清 PNG(供插入文档/打印)。

设计: 画布采用与 SVG 完全相同的坐标系(680 x 566),
      figsize 取 680/72 英寸 => 1 数据单位 == 1 pt,
      于是字号与线宽可以直接沿用 SVG 里的数值。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- 中文字体
_CHOSEN = None
for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        _CHOSEN = name
        break
    except Exception:
        continue
if _CHOSEN:
    plt.rcParams["font.sans-serif"] = [_CHOSEN]
plt.rcParams["axes.unicode_minus"] = False

W, H = 680, 566
C = dict(vcc="#C0392B", gnd="#7F8C8D", sda="#2E75B6", scl="#2E8B57", a0="#D68910",
         box="#F5F7FA", edge="#D8DEE6", esp="#E8F1FB", esp_edge="#B5D4F4",
         tx="#1F2937", tx2="#5F6B7A")

fig = plt.figure(figsize=(W / 72.0, H / 72.0), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # y 轴翻转, 与 SVG 坐标一致
ax.set_axis_off()


def line(x1, y1, x2, y2, color, lw=2.5):
    """zorder=1: 总线永远在模块方块之下 —— 视觉上被方块"截断",
    从而读成"总线依次经过每个模块"而不是"线压着文字"。"""
    ax.add_line(Line2D([x1, x2], [y1, y2], color=color, linewidth=lw,
                       solid_capstyle="butt", zorder=1))


def poly(pts, color, lw=2.0):
    """zorder=3: A0 连线要盖住方块描边(它从方块边缘出发)。"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.add_line(Line2D(xs, ys, color=color, linewidth=lw,
                       solid_capstyle="round", solid_joinstyle="round",
                       zorder=3))


def box(x, y, w, h, fill, edge, r=8, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=%g" % r,
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                zorder=2))


def txt(x, y, s, size=12, color="#1F2937", ha="center", weight="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
            fontweight=weight, zorder=5)


def dot(x, y, color, r=4.0):
    """zorder=4: 连接点画在方块之上, 正好压在方块边界上, 读成"引脚在这接了"。"""
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none",
                        zorder=4))


# ---------------------------------------------------------------- 1 总线
line(210, 104, 210, 416, C["vcc"])
line(290, 104, 290, 416, C["gnd"])
line(370, 104, 370, 342, C["sda"])
line(450, 104, 450, 342, C["scl"])

# ---------------------------------------------------------------- 2 GSR -> A0
poly([(320, 392), (505, 392), (505, 318), (470, 318)], C["a0"])
txt(514, 358, "A0", 12, "#B9770E", ha="left")

# ---------------------------------------------------------------- 3 模块方块
box(170, 146, 300, 48, C["box"], C["edge"])
box(170, 220, 300, 48, C["box"], C["edge"])
box(170, 294, 300, 48, C["box"], C["edge"])
box(180, 368, 140, 48, C["box"], C["edge"])
box(170, 36, 300, 68, C["esp"], C["esp_edge"], r=10)

# ---------------------------------------------------------------- 4 文字
txt(320, 60, "ESP32 DevKit V1", 14, C["tx"], weight="bold")
txt(320, 84, "USB 连接电脑 · COM6", 12, C["tx2"])

txt(218, 120, "3V3", 12, C["vcc"], ha="left")
txt(298, 120, "GND", 12, "#6B7378", ha="left")
txt(378, 120, "SDA · G21", 12, C["sda"], ha="left")
txt(458, 120, "SCL · G22", 12, C["scl"], ha="left")

txt(320, 162, "MAX30102", 14, C["tx"], weight="bold")
txt(320, 182, "0x57 · PPG 心率 / HRV", 12, C["tx2"])

txt(320, 236, "MPU6050", 14, C["tx"], weight="bold")
txt(320, 256, "0x68 · 运动门控（ADO 接 GND）", 12, C["tx2"])

txt(320, 310, "ADS1115", 14, C["tx"], weight="bold")
txt(320, 330, "0x48 · PGA ±4.096 V（ADDR 接 GND）", 12, C["tx2"])

txt(250, 384, "GSR", 14, C["tx"], weight="bold")
txt(250, 404, "输出 → A0", 12, C["tx2"])

txt(340, 462, "三个 I2C 器件并挂同一条总线（SDA=GPIO21、SCL=GPIO22），上拉电阻并联约 3.3 kΩ", 12, C["tx2"])
txt(340, 486, "GSR 不走总线：它的模拟输出单独接到 ADS1115 的 A0，是唯一一根非总线线", 12, C["tx2"])
txt(340, 510, "供电铁律：全部用 3.3 V、不要接 5 V；四个模块的 GND 必须共地", 12, C["tx2"])
txt(340, 534, "接线提醒：ESP32 上 G21 紧邻 GND，插线时对齐丝印字母，避免偏格", 12, C["tx2"])

# ---------------------------------------------------------------- 5 连接点
for y in (146, 220, 294):
    dot(210, y, C["vcc"]); dot(290, y, C["gnd"])
    dot(370, y, C["sda"]); dot(450, y, C["scl"])
dot(210, 368, C["vcc"])
dot(290, 368, C["gnd"])

# ---------------------------------------------------------------- 输出
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "四模态接线图.png"
fig.savefig(out, dpi=200, facecolor="white")
print("已生成:", out)
print("字体:", _CHOSEN or "(默认)")
