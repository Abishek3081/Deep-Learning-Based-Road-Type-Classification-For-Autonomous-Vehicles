import torch.multiprocessing as mp
mp.freeze_support()

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

from src.dataloader import BDD100KRoadDataset
from src.transforms import train_transform

# ---------------- CONFIG ----------------
MODEL_NAME = "efficientnet_b0"  # or mobilenet_v2 / efficientnet_b0
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True


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
    else:
        raise ValueError("Unknown model name")

    return model.to(DEVICE)


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ---------------- DATA ----------------
    train_dataset = BDD100KRoadDataset(
        images_dir=os.path.join(PROJECT_ROOT, "data", "bdd100k", "train", "images"),
        annotation_file=os.path.join(PROJECT_ROOT, "data", "bdd100k", "train", "annotations", "bdd100k_labels_images_train.json"),
        transform=train_transform
    )

    test_dataset = BDD100KRoadDataset(
        images_dir=os.path.join(PROJECT_ROOT, "data", "bdd100k", "val", "images"),
        annotation_file=os.path.join(PROJECT_ROOT, "data", "bdd100k", "val", "annotations", "bdd100k_labels_images_val.json"),
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

    # ---------------- MODEL ----------------
    model = build_model(MODEL_NAME)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # ---------------- TRAINING ----------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ---------------- TEST ----------------
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Test]"):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = running_loss / len(test_loader)
        test_acc = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    # ---------------- SAVE ----------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.save(model.state_dict(), f"models/{MODEL_NAME}.pth")

    epochs = range(1, EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_losses, label="Testing Loss")
    plt.legend()
    plt.savefig(f"results/{MODEL_NAME}_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, test_accuracies, label="Testing Accuracy")
    plt.legend()
    plt.savefig(f"results/{MODEL_NAME}_accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["City", "Highway", "Residential"])
    disp.plot(cmap="Blues")
    plt.savefig(f"results/{MODEL_NAME}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Done for {MODEL_NAME}. Results saved.")


if __name__ == "__main__":
    main()