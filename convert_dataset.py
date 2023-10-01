import os
from multiprocessing import Pool

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_full_dataset
from face_crop import crop_resize

def _load(path: str):
    img = cv2.imread(path)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _save(img: np.ndarray, path: str):
    p_img = Image.fromarray(img)
    p_img.save(path)

OUT_PATH = "datasets/final_dataset"

def convert_single(obj):
    path, label = obj
    img = _load(path)
    filename = path.split("/")[-1]
    face_image = crop_resize(img, resize_to=300)
    if label == 1:
        _save(face_image, OUT_PATH + "/living/" + filename)
    else:
        _save(face_image, OUT_PATH + "/fake/" + filename)

def convert(path: str, out: str):
    dataset = get_full_dataset(path, resize=None)
    # os.makedirs(out + "/living")
    # os.makedirs(out + "/fake")
    pool = Pool(5)
    for _ in tqdm(pool.imap_unordered(convert_single, dataset.data), total=len(dataset)):
        pass


if __name__ == "__main__":
    convert("super_dataset/final_dataset", "datasets/test_faces_dataset")