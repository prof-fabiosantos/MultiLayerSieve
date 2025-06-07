import pandas as pd
from sklearn.model_selection import GridSearchCV
from MultiLayerSieve import MultiLayerSieve
import joblib

df = pd.read_csv('pecas_treino.csv')
X = df[['comprimento', 'largura', 'peso']].values
y = df['classe'].values

clf = MultiLayerSieve(
    n_layers=2,
    feature_names=['comprimento', 'largura', 'peso'],
    n_thresholds=5   # maior granularidade = melhor ajuste
)
clf.fit(X, y)
param_grid = clf.get_param_grid()
grid = GridSearchCV(clf, param_grid, scoring='accuracy', cv=2)
grid.fit(X, y)

joblib.dump(grid.best_estimator_, 'modelo_sieve_pecas.pkl')
print("Modelo salvo como 'modelo_sieve_pecas.pkl'.")
