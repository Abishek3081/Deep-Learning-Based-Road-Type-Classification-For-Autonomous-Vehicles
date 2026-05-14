
import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)

from src.dataloader import BDD100KRoadDataset
from src.transforms import train_transform


# ================= PERFORMANCE OPTIMIZATION =================
torch.backends.cudnn.benchmark = True

# ================= PATH SETUP =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= CONFIG =================
BATCH_SIZE = 16
EPOCHS = 5          # keep same as ConvNeXt experiments
LR = 3e-4
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================


def get_dataloaders():
    train_dataset = BDD100KRoadDataset(
        images_dir=os.path.join(PROJECT_ROOT, "data/bdd100k/train/images"),
        annotation_file=os.path.join(
            PROJECT_ROOT,
            "data/bdd100k/train/annotations/bdd100k_labels_images_train.json"
        ),
        transform=train_transform
    )

    test_dataset = BDD100KRoadDataset(
        images_dir=os.path.join(PROJECT_ROOT, "data/bdd100k/val/images"),
        annotation_file=os.path.join(
            PROJECT_ROOT,
            "data/bdd100k/val/annotations/bdd100k_labels_images_val.json"
        ),
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader


def build_model(name):
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif name == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    elif name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)

    else:
        raise ValueError(f"Unknown model: {name}")

    return model.to(DEVICE)


def train_and_evaluate(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---------------- TRAINING ----------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            leave=False
        ):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # ---------------- TESTING ----------------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    return test_accuracy


def save_results_and_plot(results):
    os.makedirs("results", exist_ok=True)

    # ---------------- TABLE I (CSV) ----------------
    table_path = os.path.join("results", "table1_comparative_accuracy.csv")

    with open(table_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Testing Accuracy (%)"])
        for model, acc in results.items():
            writer.writerow([model, f"{acc*100:.2f}"])

    # ---------------- BAR GRAPH ----------------
    models = list(results.keys())
    accuracies = [acc * 100 for acc in results.values()]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(models, accuracies)

    plt.xlabel("CNN Architecture")
    plt.ylabel("Testing Accuracy (%)")
    plt.title("Comparative Performance Across CNN Architectures")
    plt.ylim(70, 85)   # stable IEEE scaling

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center",
            fontsize=9
        )

    plt.savefig(
        os.path.join("results", "fig8_comparative_bar_graph.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("✅ Table I and comparative bar graph saved in results/ folder")


def main():
    print("Using device:", DEVICE)

    train_loader, test_loader = get_dataloaders()

    models_to_test = [
        "mobilenet_v2",
        "resnet18",
        "efficientnet_b0",
        "convnext_tiny"
    ]

    results = {}

    for model_name in models_to_test:
        print(f"\n===== Training {model_name} =====")
        model = build_model(model_name)
        acc = train_and_evaluate(model, train_loader, test_loader)
        results[model_name] = acc
        print(f"{model_name} Testing Accuracy: {acc:.4f}")

    print("\n===== FINAL BENCHMARK RESULTS =====")
    for k, v in results.items():
        print(f"{k:15s}: {v*100:.2f}%")

    save_results_and_plot(results)


if __name__ == "__main__":
    main()
