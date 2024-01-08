from abc import ABC, abstractmethod
from pandas import DataFrame

TRAINING_RANDOM_STATE = 420


class IEvaluator(ABC):
    def __init__(self):
        self.hyperparameter_config = {}
        self.X_train: DataFrame = None  # Data for training
        self.y_train: DataFrame = None  # Labels for training
        self.X_test: DataFrame = None  # Data for testing
        self.y_test: DataFrame = None  # Labels for testing

    @abstractmethod
    def set_dataset(self, dataset: DataFrame):
        pass

    @abstractmethod
    def get_search_space(self, strategy) -> dict:
        pass

    @abstractmethod
    def evaluate(self, config):
        pass
