import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FraudImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir/
          train/genuine/*.jpg
          train/fraud/*.jpg
        """
        self.root = os.path.join(root_dir, split)
        self.classes = ['genuine', 'fraud']
        self.samples = []
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg','.png','.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), idx))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, path
