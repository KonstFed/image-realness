import os
import re
from random import shuffle
import glob

from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def _resize_with_ratio(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, [size, round(size * image.shape[0] / image.shape[1])])


class CustomDataset(Dataset):
    # 1 for living, 0 for fake
    def __init__(self, data_paths: list[str], labels: list[int],
                 resize: int=None, transform=None) -> None:
        assert len(data_paths) == len(labels)
        self.data = [(data_paths[i], labels[i]) for i in range(len(data_paths))]
        self._resize = resize
        self.transform = transform

    def __getitem__(self, index) -> tuple[np.ndarray, int]:
        img = Image.open(self.data[index][0])
        img = img.convert("RGB")
        if self._resize:
            img = img.resize([self._resize, self._resize], Image.Resampling.LANCZOS)

        if self.transform:
            img = self.transform(img)

        return img, self.data[index][1]

    def __len__(self):
        return len(self.data)


def get_datasets(dataset_path: str, train_test_ratio=0.1, resize=None, transform=None):
    living_names = os.listdir(os.path.join(dataset_path, "living"))
    fake_names = os.listdir(os.path.join(dataset_path, "fake"))
    limiter = min(len(living_names), len(fake_names))
    living_names = living_names[:limiter]
    fake_names = fake_names[:limiter]

    living_path = [
        os.path.join(dataset_path, "living", f)
        for f in living_names
        if re.search(r".(jpg|png|jpeg)", f) and f[0] != "."
    ]
    fake_path = [
        os.path.join(dataset_path, "fake", f)
        for f in fake_names
        if re.search(r".jpg|png|jpeg", f) and f[0] != "."
    ]
    x = [x for x in living_path] + [x for x in fake_path]
    labels = [1] * len(living_path) + [0] * len(fake_path)
    X_train, X_test, y_train, y_test = train_test_split(
        x, labels, test_size=train_test_ratio, stratify=labels
    )
    train_d = CustomDataset(X_train, y_train, resize=resize, transform=transform)
    test_d = CustomDataset(X_test, y_test, resize=resize, transform=transform)
    shuffle(train_d.data)
    shuffle(test_d.data)
    return train_d, test_d


def get_full_dataset(dataset_path: str, resize=300, transform=None):
    living_names = os.listdir(os.path.join(dataset_path, "living"))
    fake_names = os.listdir(os.path.join(dataset_path, "fake"))

    living_path = [
        os.path.join(dataset_path, "living", f)
        for f in living_names
        if re.search(r".(jpg|png|jpeg)", f) and f[0] != "."
    ]
    fake_path = [
        os.path.join(dataset_path, "fake", f)
        for f in fake_names
        if re.search(r".jpg|png|jpeg", f) and f[0] != "."
    ]
    x = [x for x in living_path] + [x for x in fake_path]
    labels = [1] * len(living_path) + [0] * len(fake_path)
    train_d = CustomDataset(x, labels, resize=resize, transform=transform)

    return train_d


if __name__ == "__main__":
    # train_d, test_d = get_datasets("validation_data")
    train_d = get_full_dataset("test_dataset")
    print(train_d[0][1])
    import matplotlib.pyplot as plt
    plt.imshow(train_d[-1][0])
    plt.show()
    print(train_d[-1][1])
