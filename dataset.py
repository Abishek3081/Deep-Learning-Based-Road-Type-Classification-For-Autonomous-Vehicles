import json

TRAIN_JSON = "../data/bdd100k/train/annotations/bdd100k_labels_images_train.json"

with open(TRAIN_JSON, "r") as f:
    data = json.load(f)

VALID_CLASSES = {"city street", "highway", "residential"}

filtered = [
    item for item in data
    if item["attributes"]["scene"] in VALID_CLASSES
]

from collections import Counter
counts = Counter(item["attributes"]["scene"] for item in filtered)

print("Filtered samples per class:")
for k, v in counts.items():
    print(k, ":", v)

print("\nTotal samples:", len(filtered))

