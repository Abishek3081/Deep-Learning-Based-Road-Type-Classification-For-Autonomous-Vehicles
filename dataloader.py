import os
import json
from PIL import Image
from torch.utils.data import Dataset

CLASS_MAP = {
    "city street": 0,
    "highway": 1,
    "residential": 2
}

class BDD100KRoadDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.samples = [
            (item["name"], CLASS_MAP[item["attributes"]["scene"]])
            for item in data
            if item["attributes"]["scene"] in CLASS_MAP
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
