import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MultiLayerSieveClassic(BaseEstimator, ClassifierMixin):
    """
    MultiLayer Sieve (Clássico): Modelo em camadas, cada uma com thresholds próprios.
    O fit não realiza ajuste automático de thresholds, somente assume os valores recebidos via parâmetro.
    Recomendado para uso com GridSearchCV externo, que otimiza thresholds em paralelo.
    """
    def __init__(self, n_layers=2, feature_names=None, n_thresholds=5, feature_ranges=None, thresholds=None):
        self.n_layers = n_layers
        self.feature_names = feature_names
        self.n_thresholds = n_thresholds
        self.feature_ranges = feature_ranges
        self.thresholds = thresholds  # Importante: fornecido externamente via grid

    def fit(self, X, y=None):
        """
        Apenas armazena nomes das features e ranges para uso no param_grid.
        Não busca thresholds automaticamente.
        """
        X = np.asarray(X)
        if self.feature_names is None:
            self.feature_names = [f'f{i}' for i in range(X.shape[1])]
        if self.feature_ranges is None:
            self.feature_ranges = {
                fname: np.linspace(X[:, i].min(), X[:, i].max(), self.n_thresholds)
                for i, fname in enumerate(self.feature_names)
            }
        # Checa se thresholds vieram do grid, senão erro amigável
        if self.thresholds is None:
            raise ValueError(
                "Você deve fornecer 'thresholds' via param_grid no GridSearchCV. "
                "Não use fit isoladamente para ajuste automático."
            )
        return self

    def predict(self, X):
        X = np.asarray(X)
        n_layers = self.n_layers
        n_samples = X.shape[0]
        classes = np.full(n_samples, n_layers)  # Classe default: n_layers (não classificado)
        for i, thres in enumerate(self.thresholds):
            mask = np.all(X < thres, axis=1)
            not_classified = (classes == n_layers)
            should_classify = mask & not_classified
            classes[should_classify] = i
        return classes

    def get_param_grid(self, X=None):
        """
        Gera um param_grid para uso externo no GridSearchCV,
        baseado nos ranges das features.
        """
        from itertools import product
        # Se feature_ranges não foi calculado ainda, precisa de X
        if self.feature_ranges is None:
            if X is None:
                raise ValueError("Forneça X para gerar os ranges das features.")
            feature_names = self.feature_names or [f'f{i}' for i in range(X.shape[1])]
            feature_ranges = {
                fname: np.linspace(X[:, i].min(), X[:, i].max(), self.n_thresholds)
                for i, fname in enumerate(feature_names)
            }
        else:
            feature_ranges = self.feature_ranges

        all_ranges = [feature_ranges[feat] for feat in (self.feature_names or feature_ranges.keys())]
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

