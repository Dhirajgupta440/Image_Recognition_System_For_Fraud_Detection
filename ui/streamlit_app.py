# ui/streamlit_app.py
import streamlit as st
import sys
import os
from PIL import Image
import io
import torch
from torchvision import transforms
import numpy as np
import cv2

# --- make src importable (adjust if your repo layout differs) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# now import project modules (your src/models/model.py should define get_resnet50)
try:
    from models.model import get_resnet50
except Exception as e:
    st.error("Couldn't import get_resnet50 from src/models/model.py. "
             "Make sure src/models/model.py exists and defines get_resnet50().\n\n" + str(e))
    raise

# optional grad-cam imports (only used if user toggles)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_CAM = True
except Exception:
    HAS_CAM = False

MODEL_PATH = os.path.join(ROOT_DIR, "models", "resnet50_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {0: "Authentic", 1: "Tampered"}

# ---- helpers ----
@st.cache_resource
def load_model(model_path=MODEL_PATH):
    """Load model (state_dict) and return model on DEVICE."""
    model = get_resnet50(num_classes=2, pretrained=False)
    state = torch.load(model_path, map_location=DEVICE)
    # If saved as dict with key 'model' or as state_dict:
    if isinstance(state, dict) and any(k.startswith("module.") or "state_dict" in k for k in state.keys()):
        # try common patterns
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            # assume it's a raw state_dict
            model.load_state_dict(state)
    else:
        try:
            model.load_state_dict(state)
        except Exception:
            # last resort: try torch.jit.load for traced model
            model = torch.jit.load(model_path, map_location=DEVICE)
            model.eval()
            return model
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image_pil(pil_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(pil_img).unsqueeze(0)

def predict(model, pil_img):
    x = preprocess_image_pil(pil_img).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        pred = int(torch.argmax(probs, dim=1).cpu().numpy()[0])
        conf = float(probs[0, pred].cpu().numpy())
    return pred, conf, x

def compute_gradcam(model, input_tensor, pred_class, target_layer):
    """Return overlayed image (numpy uint8) or raise if CAM not available."""
    if not HAS_CAM:
        raise RuntimeError("pytorch_grad_cam not installed")
    # Grad-CAM requires CPU/ CUDA config via 'device' param removed in recent versions.
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(DEVICE.type == "cuda")) \
          if "use_cuda" in GradCAM.__init__.__code__.co_varnames else GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_class)]
    # input_tensor -> tensor on DEVICE (B,C,H,W)
    grayscale_cam = cam(input_tensor=input_tensor.to(DEVICE), targets=targets)[0]
    # convert input tensor to HxWx3 float image in [0,1]
    inp = input_tensor[0].cpu()
    inp = inp.permute(1,2,0).numpy()
    # undo normalization
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = (inp * std) + mean
    inp = np.clip(inp, 0, 1)
    visualization = show_cam_on_image(inp.astype(np.float32), grayscale_cam, use_rgb=True)
    return visualization

# --- Streamlit UI ----
st.set_page_config(page_title="Fraud Detector", page_icon="🔎", layout="centered")

# Top header
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px;">
      <h1 style="display:flex">🔎 Fraud Detector</h1>
      <div style="color:gray; margin-left:8px"> — Upload an image, get prediction + explanation</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("### How to use")
st.write("Upload an image (jpg/png). The model will predict whether the image is **Authentic** or **Tampered**. "
         "Toggle **Show Grad-CAM** to see which regions influenced the prediction (if Grad-CAM is available).")

# sidebar options
st.sidebar.header("Options")
show_gradcam = st.sidebar.checkbox("Show Grad-CAM (if available)", value=True)
st.sidebar.write(f"Device: **{DEVICE.type}**")

# Upload or sample selection
col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    else:
        image = None

with col2:
    st.write("Or try a sample image:")
    sample_dir = os.path.join(ROOT_DIR, "data", "sample_images")
    # fallback to any sample_images folder used by your repo
    sample_paths = []
    if os.path.isdir(sample_dir):
        for f in os.listdir(sample_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                sample_paths.append(os.path.join(sample_dir, f))
    # show first sample if exists
    if sample_paths:
        choice = st.selectbox("Choose sample", ["-- none --"] + [os.path.basename(p) for p in sample_paths])
        if choice and choice != "-- none --":
            image = Image.open(os.path.join(sample_dir, choice)).convert("RGB")

# action
if image is None:
    st.info("Upload or select a sample image to start.")
else:
    st.image(image, caption="Input image", use_column_width=True)
    # Load model once
    try:
        model = load_model()
    except Exception as e:
        st.error("Failed to load model. Check model path and get_resnet50().\n\n" + str(e))
        st.stop()

    with st.spinner("Running inference..."):
        pred_class, confidence, input_tensor = predict(model, image)

    st.markdown("### Prediction")
    label = LABELS.get(pred_class, str(pred_class))
    st.metric(label=f"Result: {label}", value=f"{confidence*100:.2f}% confident")

    # confidence bar
    st.progress(confidence)

    # show Grad-CAM if requested
    if show_gradcam:
        if not HAS_CAM:
            st.warning("pytorch-grad-cam not installed. Install it to see Grad-CAM overlays:\n\npip install pytorch-grad-cam")
        else:
            try:
                # choose target layer for ResNet50
                # this assumes model has attribute layer4
                target_layer = model.layer4[-1]
                vis = compute_gradcam(model, input_tensor, pred_class, target_layer)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Grad-CAM overlay", use_column_width=True)
            except Exception as e:
                st.error("Grad-CAM failed: " + str(e))

    # download buttons
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    st.download_button("Download Input Image", data=buf.getvalue(), file_name="input.jpg", mime="image/jpeg")

    if show_gradcam and HAS_CAM:
        try:
            # save overlay to buffer for download
            overlay_buf = io.BytesIO()
            # vis is numpy BGR uint8
            overlay_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            overlay_img.save(overlay_buf, format="JPEG")
            st.download_button("Download Grad-CAM Overlay", data=overlay_buf.getvalue(), file_name="gradcam.jpg", mime="image/jpeg")
        except Exception:
            pass

st.markdown("---")
st.caption("Designed by Team")
