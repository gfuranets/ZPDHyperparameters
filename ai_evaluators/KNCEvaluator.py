from sklearn.neighbors import KNeighborsClassifier
from ai_evaluators.IEvaluator import IEvaluator, TRAINING_RANDOM_STATE
from sklearn.metrics import f1_score

import ray
from ray import train, tune


class KNCEvaluator(IEvaluator):
    def __init__(self):
        super().__init__()
        self.hyperparameters = {
            'n_neighbors': ([2, 3, 5, 7], (2, 10)),
            'algorithm':  [['ball_tree', 'kd_tree']], # 'algorithm': [['brute']] brute just doesn't require leaf_size and p params
            'leaf_size': ([5, 10, 25, 40], (5, 40)),
            'p': ([1, 2, 3, 4], (1, 4)),
        }

    def evaluate(self, config):
        clf = KNeighborsClassifier(
            n_neighbors=int(config['n_neighbors']),
            algorithm=config['algorithm'],
            leaf_size=int(config['leaf_size']),
            p=int(config['p'])
        )

        X_train = ray.get(config["X_train_id"])
        X_test = ray.get(config["X_test_id"])
        y_train = ray.get(config["y_train_id"])
        y_test = ray.get(config["y_test_id"])


        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1 = f1_score(y_test, predictions, average='micro')

        train.report({"f1_score": f1})