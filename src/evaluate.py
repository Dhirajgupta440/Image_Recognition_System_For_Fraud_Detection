import torch, os, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from models.model import get_resnet50
from data.dataset import FraudImageDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/resnet50_best.pth"
BATCH = 16

def evaluate(root='data'):
    model = get_resnet50(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    test_ds = FraudImageDataset(root, split='test')
    loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    print(classification_report(y_true, y_pred, target_names=['genuine','fraud']))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()

