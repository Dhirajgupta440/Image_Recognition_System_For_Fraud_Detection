import os
import shutil
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from models.model import get_resnet50
from data.dataset import FraudImageDataset

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/resnet50_best.pth"
OUT_DIR = "analysis/gradcam"


# Preprocessing
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ensure match to cam size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return img, transform(img).unsqueeze(0)


# Load model
def load_model():
    model = get_resnet50(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def run_on_folder(folder, model, target_layer):
    os.makedirs(OUT_DIR, exist_ok=True)
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"\n🔍 Running GradCAM on {len(files)} images...\n")

    cam = GradCAM(model=model, target_layers=[target_layer])

    for f in files:
        try:
            orig_img, input_tensor = preprocess_image(f)
            input_tensor = input_tensor.to(DEVICE)

            # forward pass → get predicted class
            with torch.no_grad():
                out = model(input_tensor)
                pred_class = torch.argmax(out).item()

            targets = [ClassifierOutputTarget(pred_class)]

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            # prepare original image as numpy (normalized 0-1)
            img_np = np.array(orig_img.resize((224, 224))) / 255.0

            cam_image = show_cam_on_image(img_np.astype(np.float32),
                                          grayscale_cam,
                                          use_rgb=True)

            out_name = os.path.splitext(os.path.basename(f))[0]
            save_path = os.path.join(OUT_DIR, f"{out_name}_cam.jpg")

            cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            print(f"✔ Saved: {save_path}")

        except Exception as e:
            print(f"❌ Failed on: {f}")
            print("Error:", e)

    print(f"\n✅ Grad-CAM images saved to: {OUT_DIR}")


if __name__ == "__main__":
    model = load_model()
    target_layer = model.layer4[-1]  # last conv block

    # If misclassified folder exists
    mis_dir = "analysis/misclassified"
    if os.path.exists(mis_dir):
        for sub in os.listdir(mis_dir):
            folder = os.path.join(mis_dir, sub)
            if os.path.isdir(folder):
                run_on_folder(folder, model, target_layer)
    else:
        # fallback → first 50 test images
        print("⚠ No misclassified folder → taking first 50 test images...")
        test_ds = FraudImageDataset(root_dir="data", split="test")

        # create sample folder
        tmp_folder = "analysis/sample_images"
        os.makedirs(tmp_folder, exist_ok=True)

        # copy first 50
        for i, (p, _) in enumerate(test_ds.samples[:50]):
            try:
                shutil.copy2(p, os.path.join(tmp_folder, os.path.basename(p)))
            except:
                pass

        run_on_folder(tmp_folder, model, target_layer)
