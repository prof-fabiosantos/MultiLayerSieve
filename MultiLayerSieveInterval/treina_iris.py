import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from train_and_save import fit_and_save_model

# 1. Carregar o dataset iris
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# 2. Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinar e salvar o modelo (usando sua função customizada)
# Os nomes das features são as colunas do dataframe do iris
feature_names = X.columns.tolist()
grid = fit_and_save_model(
    X_train.values, y_train.values,
    feature_names=feature_names,
    n_layers=2,      # Você pode ajustar para 3 etc, se desejar
    n_intervals=4,   # Ajuste para granularidade desejada
    scoring='accuracy',
    cv=3,
    model_path='modelo_sieve_iris.pkl',
    verbose=2,
    n_jobs=-1
)

# (Opcional) Avaliar o melhor modelo já no treino
print('Melhor score (validação cruzada):', grid.best_score_)
