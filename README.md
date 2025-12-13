# Image Recognition & Fraud Detection System using Deep Learning (ResNet50)

## 📌 Project Overview
This project is a Deep Learning–based Image Recognition System designed for accurate image classification and fraud/anomaly detection.  
It uses a ResNet50 Convolutional Neural Network trained using PyTorch and supports model interpretability using Grad-CAM, misclassification analysis, and an interactive Streamlit-based user interface.

The complete system has been developed, trained, evaluated, and tested locally in VS Code following a modular and scalable project architecture.

---

## 🧠 Model & Approach
- Architecture: ResNet50
- Framework: PyTorch
- Learning Type: Transfer Learning
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Input Image Size: 224 × 224 (RGB)
- Output: Predicted class with confidence score

---

## 📂 Project Structure
Image-Recognition-System-Project/
│
├── analysis/
│ ├── gradcam/
│ ├── misclassified/
│ └── sample_images/
│
├── api/
│ └── main.py
│
├── data/
│ └── (dataset not uploaded)
│
├── models/
│ ├── model.py
│ └── resnet50_best.pth # not uploaded (large file)
│
├── src/
│ ├── train.py
│ ├── inference.py
│ ├── evaluate.py
│ ├── gradcam.py
│ ├── gradcam_infer.py
│ ├── analysis_misclassified.py
│ └── utils.py
│
├── ui/
│ └── streamlit_app.py
│
├── Dockerfile
├── requirements.txt
├── img.jpg
└── README.md


