# MultiLayerSieveClassic.py (adaptado para peneiras não-lineares polinomiais)
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MultiLayerSieveClassic(BaseEstimator, ClassifierMixin):
    """
    MultiLayer Sieve (Clássico): Modelo em camadas, podendo ser linear ou polinomial.
    Recomenda-se uso de GridSearchCV externo para ajustar thresholds e coeficientes.
    """
    def __init__(
        self, 
        n_layers=2, 
        feature_names=None, 
        n_thresholds=5, 
        feature_ranges=None, 
        thresholds=None,
        sieve_type="linear",  # "linear" ou "poly"
        poly_coefs=None       # lista de coeficientes para polinômio de cada camada e feature
    ):
        self.n_layers = n_layers
        self.feature_names = feature_names
        self.n_thresholds = n_thresholds
        self.feature_ranges = feature_ranges
        self.thresholds = thresholds
        self.sieve_type = sieve_type
        self.poly_coefs = poly_coefs

    def fit(self, X, y=None):
        X = np.asarray(X)
        if self.feature_names is None:
            self.feature_names = [f'f{i}' for i in range(X.shape[1])]
        if self.feature_ranges is None:
            self.feature_ranges = {
                fname: np.linspace(X[:, i].min(), X[:, i].max(), self.n_thresholds)
                for i, fname in enumerate(self.feature_names)
            }
        if self.thresholds is None and self.sieve_type == "linear":
            raise ValueError(
                "Você deve fornecer 'thresholds' via param_grid no GridSearchCV para peneiras lineares."
            )
        if self.sieve_type == "poly" and self.poly_coefs is None:
            raise ValueError(
                "Para peneira polinomial, forneça poly_coefs via param_grid no GridSearchCV."
            )
        return self

    def _poly_test(self, x, coef):
        # coef deve ser [a, b, c] para cada feature: a*x^2 + b*x + c
        # x: (n_samples, n_features)
        return np.sum(coef[0] * x**2 + coef[1] * x + coef[2], axis=1)

    def predict(self, X):
        X = np.asarray(X)
        n_layers = self.n_layers
        n_samples = X.shape[0]
        classes = np.full(n_samples, n_layers)  # Default: não classificado

        if self.sieve_type == "linear":
            for i, thres in enumerate(self.thresholds):
                mask = np.all(X < thres, axis=1)
                not_classified = (classes == n_layers)
                should_classify = mask & not_classified
                classes[should_classify] = i
        elif self.sieve_type == "poly":
            for i, coef in enumerate(self.poly_coefs):
                # coef é uma matriz shape (3, n_features) para [a, b, c] de cada feature
                poly_val = self._poly_test(X, coef)
                mask = poly_val > 0  # critério de classificação, pode ser ajustado!
                not_classified = (classes == n_layers)
                should_classify = mask & not_classified
                classes[should_classify] = i
        else:
            raise ValueError("sieve_type deve ser 'linear' ou 'poly'")

        return classes

    def get_param_grid(self, X=None):
        from itertools import product
        # Linhas similares ao seu código anterior
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
        if self.sieve_type == "linear":
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
        elif self.sieve_type == "poly":
            # Gera coeficientes polinomiais a, b, c para cada feature e camada (busca simples)
            coef_range = np.linspace(-2, 2, self.n_thresholds)
            poly_coefs_list = []
            for coefs in itertools.product(coef_range, repeat=3*self.n_layers*len(all_ranges)):
                arr = np.array(coefs).reshape(self.n_layers, 3, len(all_ranges))
                poly_coefs_list.append(arr)
            return {'poly_coefs': poly_coefs_list}
        else:
            raise ValueError("sieve_type deve ser 'linear' ou 'poly'")


