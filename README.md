# Deep Learning-Based Road Type Classification for Autonomous Vehicles

## Overview
This project presents a deep learning-based road scene classification system for autonomous driving applications. The system classifies road images into three categories:

- City Street
- Highway
- Residential

The project uses transfer learning with the ConvNeXt-Tiny architecture and performs comparative analysis with ResNet18, MobileNetV2, and EfficientNet-B0 models using the BDD100K dataset.

---

## Features
- Road scene classification using deep learning
- Transfer learning with pretrained CNN models
- Comparative analysis of multiple architectures
- Accuracy and loss curve visualization
- Confusion matrix evaluation
- Grad-CAM visualization for model interpretability
- Real-world autonomous driving dataset support

---

## Models Used
- ConvNeXt-Tiny
- ResNet18
- MobileNetV2
- EfficientNet-B0

---

## Dataset
The project uses the **BDD100K (Berkeley DeepDrive 100K)** dataset.

Dataset includes:
- Urban roads
- Highways
- Residential streets
- Different weather and lighting conditions

Official Dataset:
https://bdd-data.berkeley.edu/

---

## Project Workflow

1. Input Road Image
2. Image Preprocessing
3. Feature Extraction using ConvNeXt-Tiny
4. Scene Classification
5. Output Prediction

---

## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Results
The models were evaluated using:
- Training Accuracy
- Testing Accuracy
- Training Loss
- Testing Loss
- Confusion Matrix
- Comparative Performance Analysis

### Comparative Accuracy
| Model | Accuracy |
|------|------|
| MobileNetV2 | 76.29% |
| ResNet18 | 76.54% |
| EfficientNet-B0 | 77.69% |
| ConvNeXt-Tiny | 77.24% |

---

## Folder Structure

```text

results/
models/

README.md
```

---

## Output Classes
- City Street
- Highway
- Residential

---

## Future Improvements
- Real-time video classification
- Integration with autonomous vehicle systems
- Hybrid CNN-Transformer models
- Larger dataset training
- Weather-aware classification

---

## Authors
- Abishek A 
- Harish Varadhan R
- Sanjay R
- Akshay R
- Dr.Diana Andrusia
- Dr.Mary Neebha
  Karunya University ECE(ECM-B) Final year 2022-2026

Department of Electronics and Communication Engineering

---

## References
- ConvNeXt
- EfficientNet
- ResNet
- MobileNet
- BDD100K Dataset
- Grad-CAM

---

## License
This project is developed for academic and research purposes.
