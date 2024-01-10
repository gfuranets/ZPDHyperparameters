from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler

from dataset_controllers.IDatasetController import IDatasetController
from sklearn.datasets import fetch_openml


class MNISTDatasetController(IDatasetController):
    def __init__(self):
        super().__init__()

    def prepare_dataset(self):
        dataset = fetch_openml('Fashion-MNIST', version=1, parser='auto', as_frame=True)
        df = DataFrame(dataset.data, columns=dataset.feature_names)
        df['target'] = Series(dataset.target)

        # Normalize data range (0 - 1)
        min_max_scaler = MinMaxScaler()
        x = df.values
        x_scaled = min_max_scaler.fit_transform(x)
        df = DataFrame(x_scaled)

        # Clean dataset (remove empty)
        #   Axis 0 = drop rows
        #   Axis 1 = drop columns
        #   How any = drop if 1 or more empty
        #   How all = drop if all empty
        df.dropna(axis=0, how='any', inplace=True)

        self.dataset = df