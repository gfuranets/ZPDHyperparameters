from abc import ABC, abstractmethod
from pandas import DataFrame


class IDatasetController(ABC):
    def __init__(self):
        self.dataset: DataFrame = None

    def get_dataset(self) -> DataFrame:
        return self.dataset

    @abstractmethod
    def get_sets(self):
        pass
    