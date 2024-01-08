from pandas import DataFrame

from dataset_controllers.IDatasetController import IDatasetController
from sklearn.datasets import fetch_openml


class MNISTDatasetController(IDatasetController):
    def __init__(self):
        super().__init__()

    def prepare_dataset(self):
        self.dataset = fetch_openml('mnist_784', version=1, parser='auto')
        # Clean, scale etc.
