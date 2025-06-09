import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report

# 1. Carregar o dataset iris
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# 2. Dividir em treino/teste (igual ao script de treino!)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Carregar o modelo treinado
modelo = joblib.load('modelo_sieve_iris.pkl')

# 4. Previsão
y_pred = modelo.predict(X_test.values)

# 5. Avaliação
print(classification_report(y_test, y_pred, target_names=iris.target_names))
