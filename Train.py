import pandas as pd
from sklearn.model_selection import GridSearchCV
from MultiLayerSieveClassic import MultiLayerSieveClassic
import joblib

# Ler dados de treino
df = pd.read_csv('treino.csv')
X = df[['glicose', 'pressao', 'temperatura']].values
y = df['classe_verdadeira'].values

# Instanciar o classificador
clf = MultiLayerSieveClassic(
    n_layers=2,
    feature_names=['glicose', 'pressao', 'temperatura'],
    n_thresholds=5  # granularidade dos thresholds
)
clf.fit(X, y)                # Gera os ranges automaticamente!
param_grid = clf.get_param_grid()
grid = GridSearchCV(clf, param_grid, scoring='accuracy', cv=2)
grid.fit(X, y)

# Persistir o modelo treinado
joblib.dump(grid.best_estimator_, 'modelo_sieve.pkl')

