import os

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange

from dataset import get_full_dataset
from face_crop import crop_resize

def _load(path: str):
    img = cv2.imread(path)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save(img: np.ndarray, path: str):
    p_img = Image.fromarray(img)
    p_img.save(path)
    

def convert(path: str, out: str):
    dataset = get_full_dataset(path, resize=None)
    # os.makedirs(out + "/living")
    # os.makedirs(out + "/fake")

    for i in trange(len(dataset)):
        image = _load(dataset.data[i][0])
        filename = dataset.data[i][0].split("/")[-1]
        face_image = crop_resize(image, resize_to=300)
        if dataset.data[i][1] == 1:
            save(face_image, out + "/living/" + filename)
        else:
            save(face_image, out + "/fake/" + filename)

if __name__ == "__main__":
    convert("datasets/test_dataset", "datasets/test_faces_dataset")