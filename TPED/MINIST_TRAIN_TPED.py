# demo_mnist_tped_train.py
# Train a simple MNIST classifier with TPED augmentation.
# Requirements: torch, torchvision, numpy, scipy, pillow (optional)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import TPED_fast as TP  # Make sure TPED.py is in the same directory or in PYTHONPATH


class MNISTWithTPED(Dataset):
    """
    Wrap torchvision MNIST and apply TPED augmentation on-the-fly.
    """
    def __init__(self, root="./data", train=True, download=True, use_tped=True):
        self.mnist = datasets.MNIST(root=root, train=train, download=download)
        self.use_tped = use_tped

        # Basic normalization for MNIST
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil_img, label = self.mnist[idx]  # PIL image, grayscale

        if self.use_tped:
            # TPED expects PIL / numpy / torch; PIL is fine.
            # For MNIST, you may want a smaller delta_max than default.
            pil_img = TP.TPED(
                pil_img,
                bboxes=None,
                sigma=0.10,       # relative to min(H,W)=28
                delta_max=0.05,   # safer for digits than 0.10
                is_fold_free=True,
                max_tries=10,
                return_info=False,
            )

        x = self.to_tensor(pil_img)     # [1, 28, 28], float in [0,1]
        x = self.normalize(x)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = MNISTWithTPED(train=True, use_tped=True)
    test_ds = MNISTWithTPED(train=False, use_tped=False)  # do NOT augment test

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = SmallCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    epochs = 3
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

        train_loss, train_acc = evaluate(model, train_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {ep:02d} | train acc={train_acc:.4f} loss={train_loss:.4f} | test acc={test_acc:.4f} loss={test_loss:.4f}")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/mnist_cnn_tped.pth")
    print("Saved: ./checkpoints/mnist_cnn_tped.pth")


if __name__ == "__main__":
    main()