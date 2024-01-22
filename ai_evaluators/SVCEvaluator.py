from sklearn.svm import SVC
from ai_evaluators.IEvaluator import IEvaluator, TRAINING_RANDOM_STATE
from sklearn.metrics import f1_score

import ray
from ray import train


class SVCEvaluator(IEvaluator):
    def __init__(self):
        super().__init__()
        self.hyperparameters = {
            'C': [[0.1, 1, 10], (1e-3, 1e3)],
            'kernel': [['linear', 'rbf', 'poly'], (0,)],
            'degree': [[2, 3, 5], (2, 5)],
            'gamma': [[0.1, 1, 10], (1e-4, 1e2)]
        }

    def evaluate(self, config):
        
        clf = SVC(
            C=config["C"],
            kernel=config["kernel"],
            degree=config["degree"],
            gamma=config["gamma"]
        )

        X_train = ray.get(config["X_train_id"])
        X_test = ray.get(config["X_test_id"])
        y_train = ray.get(config["y_train_id"])
        y_test = ray.get(config["y_test_id"])


        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1 = f1_score(y_test, predictions, average='micro')

        train.report({"f1_score": f1})