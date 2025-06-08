# learnable_sieve.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MultiLayerSieveClassic(BaseEstimator, ClassifierMixin):
    """
    Classificador tipo peneira multicamadas.
    Gera os ranges automaticamente se não forem informados pelo usuário.
    """
    def __init__(
        self,
        n_layers=2,
        feature_names=None,
        thresholds=None,
        feature_ranges=None,  # dict: feature_name -> array de valores de threshold (opcional)
        n_thresholds=10       # nº de valores no range (se gerar automaticamente)
    ):
        self.n_layers = n_layers
        self.feature_names = feature_names
        self.thresholds = thresholds
        self.feature_ranges = feature_ranges
        self.n_thresholds = n_thresholds

    def fit(self, X, y=None):
        # Se feature_ranges não foi fornecido, gera automaticamente
        if self.feature_ranges is None:
            if self.feature_names is None:
                raise ValueError("feature_names deve ser informado para geração automática dos ranges.")
            self.feature_ranges = {
                name: np.linspace(X[:, i].min(), X[:, i].max(), self.n_thresholds)
                for i, name in enumerate(self.feature_names)
            }
        return self

    def predict(self, X):
        X = np.asarray(X)
        n_layers = self.n_layers
        n_samples = X.shape[0]
        classes = np.full(n_samples, n_layers)  # Classe default: quem passa por todas as peneiras
        for i, layer_thres in enumerate(self.thresholds):
            mask = np.all(X < layer_thres, axis=1)
            not_classified = (classes == n_layers)
            should_classify = mask & not_classified
            classes[should_classify] = i
        return classes

    def get_param_grid(self):
        """
        Gera o param_grid para GridSearchCV/RandomizedSearchCV automaticamente,
        com base em feature_ranges e número de camadas.
        """
        # Checa se já existe feature_ranges, senão lança erro
        if (self.feature_names is None) or (self.feature_ranges is None):
            raise ValueError("Defina feature_names e (feature_ranges ou treine com .fit()) antes de gerar o grid.")

        from itertools import product
        all_ranges = [self.feature_ranges[feat] for feat in self.feature_names]
        layer_thresholds = [list(product(*all_ranges)) for _ in range(self.n_layers)]

        if self.n_layers == 2:
            thresholds_list = [
                [np.array(t1), np.array(t2)]
                for t1 in layer_thresholds[0]
                for t2 in layer_thresholds[1]
            ]
        else:
            import itertools
            thresholds_list = [
                [np.array(x) for x in comb]
                for comb in itertools.product(*layer_thresholds)
            ]
        return {'thresholds': thresholds_list}

