from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange

from depth import Model
from dataset import get_full_dataset, CustomDataset

def test(model: Model, test_d: CustomDataset):
    predictions = []
    labels = []
    for i in trange(len(test_d)):
        img, label = test_d[i]
        predicted = model.forward(img)
        predictions.append(predicted)
        labels.append(label)
        del img
    print("accuracy", accuracy_score(labels, predictions))
    print("f1-score", f1_score(labels, predictions))

if __name__ == "__main__":
    model = Model(load_path="weights/svc.joblib")
    test_d = get_full_dataset("validation_data")
    test(model, test_d)