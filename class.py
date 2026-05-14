import os
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# ---------------- PATHS ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "convnext_scene.pth")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "val", "images")

CLASSES = ["City Street", "Highway", "Residential"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD MODEL ----------------
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, 3)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# ---------------- FIND ONE IMAGE PER CLASS ----------------
found = {0: None, 1: None, 2: None}

image_files = os.listdir(TEST_IMAGES_DIR)

for img_name in image_files:
    if all(found.values()):
        break

    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        continue

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred = outputs.argmax(dim=1).item()

    if found[pred] is None:
        found[pred] = (img, pred)
        print(f"Found example for class: {CLASSES[pred]}")

# ---------------- PLOT RESULTS ----------------
plt.figure(figsize=(12, 4))

idx = 1
for class_id in range(3):
    if found[class_id] is not None:
        img, pred = found[class_id]
        plt.subplot(1, 3, idx)
        plt.imshow(img)
        plt.title(f"Predicted: {CLASSES[pred]}")
        plt.axis("off")
        idx += 1

plt.suptitle("Visual Output of Road Type Classification", fontsize=16)
plt.tight_layout()
plt.show()
