import matplotlib.pyplot as plt

# ---------------- DATA ----------------
models = [
    "MobileNet-V2",
    "ResNet-18",
    "EfficientNet-B0",
    "ConvNeXt-Tiny"
]

accuracies = [76.29, 76.54, 77.69, 77.24]

# ---------------- PLOT ----------------
plt.figure(figsize=(6, 4))

bars = plt.bar(models, accuracies)

# Labels
plt.xlabel("CNN Architecture")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy Comparison Across CNN Architectures")

# Value labels on top of bars
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"{acc:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

# Y-axis range for better visibility
plt.ylim(75, 80)

# Save figure
plt.tight_layout()
plt.savefig(
    "results/fig8_accuracy_comparison_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✅ Bar graph saved as fig8_accuracy_comparison_bar.png")
