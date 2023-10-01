import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from depth import Model
from train import get_vgg
from face_crop import crop_resize_deploy


class VGG(Model):
    def __init__(self, weights_path: str, confidence: float = 0.5) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.confidence = confidence
        self.model = get_vgg()
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(weights_path))
        else:
            self.model.load_state_dict(
                torch.load(weights_path, map_location=torch.device("cpu"))
            )
        self.model.to(self.device)

    def batch_forward(self, images: list[Image.Image]) -> list[int]:
        empty_indx = []
        cropped = []
        for i, image in enumerate(images):
            c_cropped = crop_resize_deploy(np.array(image))
            if c_cropped is None:
                empty_indx.append(i)
            else:
                c_cropped = Image.fromarray(c_cropped)
                c_cropped = self.transform(c_cropped)
                cropped.append(c_cropped)
        if len(cropped) == 0:
            return [-1 for _ in range(len(empty_indx))]
        batch = torch.stack(cropped)
        batch = batch.to(self.device)
        predictions = [-2 for _ in range(len(images))]
        for empty_idx in empty_indx:
            predictions[empty_idx] = -1
        
        model_out = self.model(batch).view(-1)
        labels = (model_out > self.confidence).tolist()
        labels = list(map(lambda x: int(x), labels))
        l_i = 0
        for i in range(len(predictions)):
            if predictions[i] == -1:
                continue
            predictions[i] = labels[l_i]
            l_i += 1
        assert -2 not in predictions
        return predictions

    def forward(self, image: Image.Image | list[Image.Image]) -> int:

        image = crop_resize_deploy(np.array(image))
        if image is None:
            return -1
        self.cropped_img_ = image.copy()
        image = Image.fromarray(image)
        image = self.transform(image)
        image = torch.reshape(image, [1] + [*image.shape])
        is_real = self.model(image) > self.confidence
        if is_real:
            return 1
        else:
            return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = VGG("weights/best.pt")
    data = [Image.open("test_images/3d_mask.jpg"), Image.open("test_images/real.jpg")]
    print(model.batch_forward(data[:1]))
    