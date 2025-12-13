import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.model import get_resnet50
from data.dataset import FraudImageDataset
from utils import save_model
from tqdm import tqdm
import os

def train(root_data='data', epochs=10, batch_size=16, lr=1e-4, save_path='models/resnet50_best.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet50(num_classes=2, pretrained=True).to(device)

    train_ds = FraudImageDataset(root_data, split='train')
    val_ds = FraudImageDataset(root_data, split='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train(root_data='data', epochs=8, batch_size=16, lr=1e-4, save_path='models/resnet50_best.pth')
