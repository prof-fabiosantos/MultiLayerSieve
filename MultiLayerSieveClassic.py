# MultiLayerSieveClassic.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MultiLayerSieveClassic(BaseEstimator, ClassifierMixin):
    def __init__(self, n_layers=2, feature_names=None, n_thresholds=5, feature_ranges=None, thresholds=None):
        self.n_layers = n_layers
        self.feature_names = feature_names
        self.n_thresholds = n_thresholds
        self.feature_ranges = feature_ranges
        self.thresholds = thresholds  # <-- importante

    def fit(self, X, y=None):
        if self.feature_names is None:
            self.feature_names = [f'f{i}' for i in range(X.shape[1])]
        if self.feature_ranges is None:
            self.feature_ranges = {
                fname: np.linspace(X[:, i].min(), X[:, i].max(), self.n_thresholds)
                for i, fname in enumerate(self.feature_names)
            }
        # Só ajuste/otimize thresholds se não vierem prontos do grid
        if self.thresholds is None:
            # gere thresholds default (você pode deixar vazio ou raise erro se quiser)
            # Mas como você está usando GridSearch, sempre vai vir no param_grid
            pass
        return self

    def predict(self, X):
        X = np.asarray(X)
        n_layers = self.n_layers
        n_samples = X.shape[0]
        classes = np.full(n_samples, n_layers)
        for i, thres in enumerate(self.thresholds):
            mask = np.all(X < thres, axis=1)
            not_classified = (classes == n_layers)
            should_classify = mask & not_classified
            classes[should_classify] = i
        return classes

    def get_param_grid(self):
        from itertools import product
        all_ranges = [self.feature_ranges[feat] for feat in self.feature_names]
        layer_thresholds = [list(product(*all_ranges)) for _ in range(self.n_layers)]
        import itertools
        if self.n_layers == 2:
            thresholds_list = [
                [np.array(t1), np.array(t2)]
                for t1 in layer_thresholds[0]
                for t2 in layer_thresholds[1]
            ]
        else:
            thresholds_list = [
                [np.array(x) for x in comb]
                for comb in itertools.product(*layer_thresholds)
            ]
        return {'thresholds': thresholds_list}
