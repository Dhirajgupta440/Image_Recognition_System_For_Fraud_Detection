import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# NOTE: For production use consider the 'grad-cam' PyPI package or Captum.
# This is a simple implementation for visualization.

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)  # 1,3,224,224

def overlay_heatmap(orig_img_path, heatmap, alpha=0.5):
    img = cv2.imread(orig_img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return overlay
