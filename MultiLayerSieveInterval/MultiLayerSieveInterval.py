# MultiLayerSieveInterval.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import product
from sklearn.preprocessing import PolynomialFeatures

class MultiLayerSieveInterval(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_layers=2,
        n_intervals=2,
        feature_names=None,
        thresholds=None,
        non_linear_degree=1
    ):
        """
        n_layers: número de camadas (peneiras)
        n_intervals: granularidade dos thresholds
        feature_names: nomes das features
        thresholds: thresholds para cada camada (definidos via GridSearchCV)
        non_linear_degree: grau da transformação polinomial (1=linear)
        """
        self.n_layers = n_layers
        self.n_intervals = n_intervals
        self.feature_names = feature_names
        self.thresholds = thresholds
        self.non_linear_degree = non_linear_degree

    def fit(self, X, y=None):
        """
        Armazena apenas info das features.
        Os thresholds são definidos via param_grid do GridSearchCV.
        """
        X = np.asarray(X)
        if self.feature_names is None:
            self.feature_names = [f'f{i}' for i in range(X.shape[1])]
        # Thresholds NUNCA devem ser gerados automaticamente aqui, apenas via GridSearchCV.
        if self.thresholds is None:
            raise ValueError(
                "Você deve fornecer 'thresholds' via param_grid no GridSearchCV. "
                "Não use fit isoladamente para ajuste automático."
            )
        return self

    def _transform_features(self, X):
        """
        Aplica transformação polinomial se necessário.
        """
        if self.non_linear_degree == 1:
            return X
        else:
            poly = PolynomialFeatures(self.non_linear_degree, include_bias=False)
            return poly.fit_transform(X)

    def predict(self, X):
        """
        Aplica as peneiras na ordem, classificando cada amostra na primeira peneira que passar.
        """
        X = np.asarray(X)
        Xp = self._transform_features(X)
        n_layers = self.n_layers
        n_samples = X.shape[0]
        classes = np.full(n_samples, n_layers)  # Não classificado
        for i, (mins, maxs) in enumerate(self.thresholds):
            mask = np.all((Xp > mins) & (Xp < maxs), axis=1)
            not_classified = (classes == n_layers)
            should_classify = mask & not_classified
            classes[should_classify] = i
        return classes

    def get_param_grid(self, X):
        """
        Gera todos os thresholds possíveis para GridSearchCV.
        X deve ser os dados originais.
        """
        X = np.asarray(X)
        Xp = self._transform_features(X)
        n_features = Xp.shape[1]
        # Cria ranges para cada feature transformada
        feature_ranges = {
            i: np.linspace(Xp[:, i].min(), Xp[:, i].max(), self.n_intervals)
            for i in range(n_features)
        }
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

