import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import convnext_tiny
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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
        torch.load(
            os.path.join(PROJECT_ROOT, "models/convnext_scene.pth"),
            map_location=device
        )
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

    image, label = val_dataset[0]   # pick one clean validation image
    input_tensor = image.unsqueeze(0).to(device)

    # ---------------- GRAD-CAM ----------------
    # ConvNeXt last feature block
    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    # ---------------- VISUALIZATION ----------------
    rgb_img = image.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

    heatmap = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title("Grad-CAM Visualization (ConvNeXt-Tiny)")
    plt.savefig(
        "fig7_gradcam_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Grad-CAM heatmap saved as fig7_gradcam_heatmap.png")


if __name__ == "__main__":
    main()
