from joblib import dump, load


from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cv2


from face_crop import crop_face
from dataset import CustomDataset, get_datasets


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(
            image, self.numPoints, self.radius, method="uniform"
        )

        (hist, _) = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2),
        )

        hist = hist.astype("float")
        hist /= hist.sum() + eps
        return hist

class Model:
    def forward(self, image: np.ndarray) -> int:
        raise NotImplementedError
    
    def train(self, train_data: CustomDataset) -> None:
        raise NotImplementedError
    
    def save(self, path: str = 'model.joblib'):
        dump(self.model, path)

    def load(self, path: str = 'model.joblib'):
        self.model = load(path)

class SVC(Model):
    def __init__(self, load_path: str = None) -> None:
        self.desc = LocalBinaryPatterns(24, 8)
        if load_path:
            super().load(load_path)
        else:
            self.model = LinearSVC(C=100, max_iter=10000)

    def _hist(self, image: np.ndarray):
        out = crop_face(image)
        self.cropped_face_ = out.copy()
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        return self.desc.describe(out)

    def _extract_face(self, data: CustomDataset):
        hists = []
        labels = []
        for i in range(len(data)):
            x, y = data[i]
            hist = self._hist(x)
            hists.append(hist)
            labels.append(y)

        return hists, labels

    def train(self, train_data: CustomDataset):
        hists_train, labels_train = self._extract_face(train_data)
        self.model.fit(hists_train, labels_train)

    def forward(self, image: np.ndarray) -> int:
        hist = self._hist(image)
        return self.model.predict([hist])

class RandomForest(SVC):
    def __init__(self, load_path: str = None) -> None:
        if load_path:
            super().load()
        else:
            self.model = RandomForestClassifier()

if __name__ == "__main__":
    m = RandomForest()
    train_d, test_d = get_datasets("validation_data")
    m.train(train_d)
    # m.save("backend/weights/svc.joblib")
    i = 0
    predicted = m.forward(test_d[i][0])
    print("predicted", predicted)
    print("real", test_d[i][1])
    import matplotlib.pyplot as plt
    plt.imshow(train_d[i][0])
    plt.show()