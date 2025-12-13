# src/analysis_misclassified.py
import os
import shutil
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from models.model import get_resnet50
from data.dataset import FraudImageDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/resnet50_best.pth"
BATCH = 16
OUT_DIR = "analysis/misclassified"

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = get_resnet50(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    test_ds = FraudImageDataset(root_dir='data', split='test')
    loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    y_true, y_pred = [], []
    paths = []
    with torch.no_grad():
        for imgs, labels, img_paths in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
            paths.extend(img_paths)

    # Save classification report and confusion matrix
    print(classification_report(y_true, y_pred, target_names=['genuine','fraud']))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # copy misclassified images into folders for review
    for p, true, pred in zip(paths, y_true, y_pred):
        if true != pred:
            subdir = f"{OUT_DIR}/{ 'true_'+str(true) + '_pred_'+str(pred) }"
            os.makedirs(subdir, exist_ok=True)
            try:
                shutil.copy2(p, os.path.join(subdir, os.path.basename(p)))
            except Exception as e:
                print("Failed copying", p, e)

    print(f"Misclassified images copied to {OUT_DIR}")

if __name__ == "__main__":
    run()
