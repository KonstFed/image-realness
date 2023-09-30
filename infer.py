import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from depth import Model
from train import get_vgg
from face_crop import crop_resize


class VGG(Model):
    def __init__(self, weights_path:str, confidence: float = 0.5) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    def forward(self, image: Image) -> bool:
        image = crop_resize(np.array(image))
        image = Image.fromarray(image)
        image = self.transform(image)
        image = torch.reshape(image, [1] + [*image.shape])
        return self.model(image) > self.confidence
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model = VGG("weights/best.pt")
    


