from dataset_controllers.IDatasetController import IDatasetController

from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from pandas import DataFrame

class ForestDatasetController(IDatasetController):
    def __init__(self):
        super().__init__()
        dataset = fetch_covtype(as_frame = True)

        dataset["target"] = dataset["target"].astype(np.uint8)

        # Scaling data
        scaler = MinMaxScaler()
        dataset["data"] = DataFrame(scaler.fit_transform(dataset["data"].astype(np.float64)))
        self.dataset = dataset

    def get_sets(self, train_size = 0.8):
        X, y = self.dataset["data"], self.dataset["target"]
        num_samples = len(X)
        train_end = int(num_samples * train_size)
        
        X_train, X_test = X[:train_end], X[train_end:]
        y_train, y_test = y[:train_end], y[train_end:]
        
        return X_train, X_test, y_train, y_test