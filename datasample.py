import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Get project root (PythonProjectF)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct dataset path
BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "bdd100k", "train", "images"
)

print("Looking for images in:", BASE_DIR)

# Safety check
assert os.path.exists(BASE_DIR), "Dataset path not found!"

# Pick random images
image_files = random.sample(os.listdir(BASE_DIR), 6)

plt.figure(figsize=(10, 6))

for i, img_name in enumerate(image_files):
    img_path = os.path.join(BASE_DIR, img_name)
    img = Image.open(img_path).convert("RGB")

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")


plt.tight_layout()
plt.savefig("fig3_dataset_samples.png", dpi=300)
plt.show()
