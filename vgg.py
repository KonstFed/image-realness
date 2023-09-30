import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms


from dataset import CustomDataset, get_datasets, get_full_dataset
from depth import Model
from tests import test

BATCH_SIZE = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# model = torch.hub.load("pytorch/vision:v0.10.0", "vgg11_bn", pretrained=True)


def get_model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True)

    for param in model.features.parameters():
        param.require_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 1000),
        nn.ReLU(inplace=True),
        nn.Linear(1000, 1),
        nn.Sigmoid()
    )
    return model


class VGG(Model):
    def __init__(self) -> None:
        self.model = get_model()

    def forward(self, image: np.ndarray) -> int:
        return self.image


def get_loaders():
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    )
    train_d, val_d = get_datasets("datasets/test_dataset", train_test_ratio=0.9, transform=transform)
    def collate(batch):
        img, label = zip(*batch)
        img = torch.stack(img)
        img = img.to(device)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.view(BATCH_SIZE, 1)
        label = label.to(device)
        return img, label

    train_loaders = DataLoader(
        train_d, num_workers=2, batch_size=BATCH_SIZE, collate_fn=collate
    )
    val_loaders = DataLoader(
        val_d, num_workers=2, batch_size=BATCH_SIZE, collate_fn=collate
    )
    return train_loaders, val_loaders

def train_epoch(model, loader, optimizer, loss):
    model.train()
    bar = tqdm(enumerate(loader, start=1))
    all_loss = 0
    for i, (img, target) in bar:
        bar.set_description(f"Epoch {i}")
        predictions = model(img)
        optimizer.zero_grad()
        c_loss = loss(predictions, target)
        c_loss.backward()
        optimizer.step()

        all_loss += c_loss.item()

        bar.set_postfix({"loss": all_loss / i})
    

def train(model):
    loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    epoch = 10
    train_l, val_l = get_loaders()
    for i in range(epoch):
        train_epoch(model, train_l, optimizer, loss)

if __name__ == "__main__":
    train(get_model())