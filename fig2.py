import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_box(ax, xy, width, height, text):
    rect = Rectangle(xy, width, height, fill=False, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        wrap=True
    )

# ---------------- FIGURE ----------------
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")

# IMPORTANT: set manual limits (this fixes clipping)
ax.set_xlim(0, 7.6)
ax.set_ylim(0, 1.5)

# Box geometry
y = 0.6
h = 0.35
w = 1.3

draw_box(ax, (0.1, y), w, h, "Input Image\n(224×224 RGB)")
draw_box(ax, (1.6, y), w, h, "Preprocessing\nResize + Normalize")
draw_box(ax, (3.1, y), w, h, "ConvNeXt-Tiny\nFeature Extractor")
draw_box(ax, (4.6, y), w, h, "Global Avg Pool\n+ Fully Connected")
draw_box(ax, (6.1, y), w, h, "Scene Output\nCity | Highway | Residential")

# Arrows
for x in [1.4, 2.9, 4.4, 5.9]:
    ax.annotate(
        "",
        xy=(x + 0.2, y + h / 2),
        xytext=(x, y + h / 2),
        arrowprops=dict(arrowstyle="->", linewidth=1.2)
    )

# Save
plt.savefig("fig2_architecture.png", dpi=300, bbox_inches="tight")
plt.show()
