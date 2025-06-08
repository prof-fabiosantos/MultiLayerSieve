# MultiLayerSieveInterval.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MultiLayerSieveInterval(BaseEstimator, ClassifierMixin):
    def __init__(self, n_layers=2, n_intervals=2, feature_names=None, thresholds=None):
        self.n_layers = n_layers
        self.n_intervals = n_intervals
        self.feature_names = feature_names
        self.thresholds = thresholds

    def fit(self, X, y):
        # Não faz grid search aqui! Apenas armazena X, y se quiser.
        # thresholds deve ser definido externamente pelo GridSearchCV.
        return self

    def predict(self, X):
        X = np.asarray(X)
        n_layers = self.n_layers
        n_samples = X.shape[0]
        classes = np.full(n_samples, n_layers)  # Default: não classificado
        for i, (mins, maxs) in enumerate(self.thresholds):
            mask = np.all((X > mins) & (X < maxs), axis=1)
            not_classified = (classes == n_layers)
            should_classify = mask & not_classified
            classes[should_classify] = i
        return classes

    def get_param_grid(self, X):
        n_features = X.shape[1]
        feature_ranges = {
            i: np.linspace(X[:, i].min(), X[:, i].max(), self.n_intervals)
            for i in range(n_features)
        }
        # Construa todas as combinações possíveis de intervalos por camada igual ao seu fit antigo
        from itertools import product
        intervals_per_feature = []
        for i in range(n_features):
            vals = feature_ranges[i]
            intervals = [(a, b) for a in vals for b in vals if a < b]
            intervals_per_feature.append(intervals)
        all_intervals = list(product(*intervals_per_feature))
        all_layer_combinations = list(product(all_intervals, repeat=self.n_layers))
        thresholds_list = []
        for layers_intervals in all_layer_combinations:
            camada_thresholds = []
            for t in layers_intervals:
                mins, maxs = zip(*t)
                camada_thresholds.append((np.array(mins), np.array(maxs)))
            thresholds_list.append(camada_thresholds)
        return {'thresholds': thresholds_list}
