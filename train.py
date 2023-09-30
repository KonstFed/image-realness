from depth import Model

from dataset import CustomDataset
from tests import test

def train_depth(train_d: CustomDataset):
    model = Model()
    model.train(train_d)
    model.save("weights/current.joblib")

