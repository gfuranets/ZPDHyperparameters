from pandas import DataFrame

from dataset_controllers.IDatasetController import IDatasetController
from sklearn.datasets import load_iris


class IRISDatasetController(IDatasetController):
    def __init__(self):
        super().__init__()

    def prepare_dataset(self):
        self.dataset = load_iris()
        # Clean, scale etc.
