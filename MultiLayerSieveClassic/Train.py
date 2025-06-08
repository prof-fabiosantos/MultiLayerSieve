import pandas as pd
from train_and_save import fit_and_save_model

# 1. Carregar os dados
df = pd.read_csv('treino.csv')
X = df[['glicose', 'pressao', 'temperatura']].values
y = df['classe_verdadeira'].values

# 2. Rodar todo o processo
grid = fit_and_save_model(
    X, y,
    feature_names=['glicose', 'pressao', 'temperatura'],
    n_layers=2,
    n_thresholds=5,
    scoring='accuracy',
    cv=2,
    model_path='modelo_sieve.pkl',
    verbose=2
)


