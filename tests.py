from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange

from infer import VGG
from depth import Model
from dataset import get_full_dataset, CustomDataset

def test(model: Model, test_d: CustomDataset):
    predictions = []
    labels = []
    for i in trange(len(test_d)):
        img, label = test_d[i]
        predicted = model.forward(img)
        if predicted == -1:
            print("Kek")
            predicted = 0
        predictions.append(predicted)
        labels.append(label)
        del img
    print("accuracy", accuracy_score(labels, predictions))
    print("f1-score", f1_score(labels, predictions))

if __name__ == "__main__":
    model = VGG(weights_path="weights/best.pt")
    test_d = get_full_dataset("datasets/validation_data")
    test(model, test_d)