from sklearn.preprocessing import StandardScaler

import numpy as np
from pandas import DataFrame

from dataset_controllers.IDatasetController import IDatasetController
from sklearn.datasets import fetch_openml


class MNISTFashionDatasetController(IDatasetController):
    def __init__(self):
        super().__init__()
        dataset = fetch_openml('Fashion-MNIST', version=1, parser='auto', as_frame=True)
        # Converting labels from str to int
        dataset["target"] = dataset["target"].astype(np.uint8)

        # Scaling data
        scaler = StandardScaler()
        dataset["data"] = DataFrame(scaler.fit_transform(dataset["data"].astype(np.float64)))
        self.dataset = dataset

    def get_sets(self):
        X, y = self.dataset["data"], self.dataset["target"]
        return X[:60000], X[60000:], y[:60000], y[60000:]
    