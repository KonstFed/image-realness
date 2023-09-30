import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score

from dataset import CustomDataset, get_datasets, get_full_dataset
from depth import Model
from bor20 import LivenessNet

import warnings

warnings.filterwarnings("ignore")

BATCH_SIZE = 32
EPOCH = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# model = torch.hub.load("pytorch/vision:v0.10.0", "vgg11_bn", pretrained=True)


def get_vgg():
    model = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True)

    # for param in model.features.parameters():
    #     param.require_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 1),
        nn.Sigmoid(),
    )
    return model


class VGG(Model):
    def __init__(self) -> None:
        self.model = get_vgg()

    def forward(self, image: np.ndarray) -> int:
        return self.image


def collate(batch):
    img, label = zip(*batch)
    img = torch.stack(img)
    img = img.to(device)
    label = torch.tensor(label, dtype=torch.float32)
    label = label.view(label.shape[0], 1)
    label = label.to(device)
    return img, label


def get_loaders():
    transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_d = get_full_dataset("datasets/test_faces_dataset", transform=transform)
    val_d = get_full_dataset("datasets/validation_faces_dataset", transform=transform)

    train_loaders = DataLoader(
        train_d, num_workers=2, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True
    )
    val_loaders = DataLoader(
        val_d, num_workers=2, batch_size=BATCH_SIZE, collate_fn=collate
    )
    return train_loaders, val_loaders


def train_epoch(model, loader, optimizer, loss, epoch_number):
    model.train()
    bar = tqdm(loader)
    all_loss = 0
    for i, (img, target) in enumerate(bar, start=1):
        bar.set_description(f"Epoch {epoch_number}")
        predictions = model(img)
        optimizer.zero_grad()
        c_loss = loss(predictions, target)
        c_loss.backward()
        optimizer.step()

        all_loss += c_loss.item()

        bar.set_postfix({"loss": all_loss / i})


def count_metrics(model, loader, loss):
    model.eval()
    bar = tqdm(loader)
    all_loss = 0
    predicted = []
    actual = []
    with torch.no_grad():
        for i, (img, target) in enumerate(bar, start=1):
            bar.set_description(f"Validating")
            predictions = model(img)
            c_loss = loss(predictions, target)

            all_loss += c_loss.item()
            predicted += torch.round(predictions).tolist()
            actual += target.tolist()

            bar.set_postfix({"loss": all_loss / i})
    f1 = f1_score(actual, predicted)
    acc = accuracy_score(actual, predicted)
    print(f"Validation\nf1: {f1}, acc: {acc}")
    return f1


def train(model):
    model.to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_l, val_l = get_loaders()
    best_score = -np.inf
    for i in range(EPOCH):
        train_epoch(model, train_l, optimizer, loss, i + 1)
        score = count_metrics(model, val_l, loss)
        if score > best_score:
            torch.save(model.state_dict(), "weights/best.pt")
            best_score = score
        scheduler.step()
    print("Best f1-score: ", best_score)

if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    model = get_vgg()
    train(model)
