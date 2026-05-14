import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import convnext_tiny

from src.dataloader import BDD100KRoadDataset
from src.transforms import train_transform


def main():
    # ---------------- PATH SETUP ----------------
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- LOAD MODEL ----------------
    model = convnext_tiny()
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3)
    model.load_state_dict(
        torch.load(os.path.join(PROJECT_ROOT, "models/convnext_scene.pth"),
                   map_location=device)
    )
    model.to(device)
    model.eval()

    # ---------------- LOAD DATA ----------------
    val_dataset = BDD100KRoadDataset(
        images_dir=os.path.join(PROJECT_ROOT, "data/bdd100k/val/images"),
        annotation_file=os.path.join(
            PROJECT_ROOT,
            "data/bdd100k/val/annotations/bdd100k_labels_images_val.json"
        ),
        transform=train_transform
    )

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # ---------------- PREDICTION ----------------
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ---------------- CONFUSION MATRIX ----------------
    class_names = ["City Street", "Highway", "Residential"]
    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix for Road Scene Classification")
    plt.savefig("fig6_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix saved as fig6_confusion_matrix.png")


if __name__ == "__main__":
    main()
