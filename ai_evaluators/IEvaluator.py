from abc import ABC, abstractmethod
from pandas import DataFrame

TRAINING_RANDOM_STATE = 420


class IEvaluator(ABC):
    def __init__(self):
        self.hyperparameter_config = {}

    @abstractmethod
    def get_search_space(self, strategy) -> dict:
        pass

    @abstractmethod
    def evaluate(self, config, X_train, X_test, y_train, y_test):
        pass
