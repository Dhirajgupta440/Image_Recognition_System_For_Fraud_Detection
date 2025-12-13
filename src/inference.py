import torch
import argparse
from PIL import Image
from torchvision import transforms
import os
import sys

# import your model builder
from models.model import get_resnet50  


def load_model(model_path, device):
    """
    Loads either:
      - a regular PyTorch model saved with state_dict
      - OR a TorchScript traced model (.pt)
    """
    if model_path.endswith(".pt") or model_path.endswith(".pth"):
        try:
            # Try TorchScript first
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
            print("[INFO] Loaded TorchScript model.")
            return model
        except Exception:
            print("[INFO] Loading regular PyTorch model (state_dict).")
    
    # Normal PyTorch model
    model = get_resnet50(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """
    Applies the same transforms used during training
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)


def predict(model, img_tensor, device):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Image Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, default="models/resnet50_best.pth",
                        help="Path to trained model")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading model...")
    model = load_model(args.model_path, device)

    print("[INFO] Pre-processing image...")
    img_tensor = preprocess_image(args.image)

    print("[INFO] Running inference...")
    pred_class, confidence = predict(model, img_tensor, device)

    label_map = {
        0: "Authentic",
        1: "Tampered"
    }

    print("\n====================")
    print(f"Prediction : {label_map.get(pred_class, pred_class)}")
    print(f"Confidence : {confidence:.4f}")
    print("====================\n")


if __name__ == "__main__":
    main()
