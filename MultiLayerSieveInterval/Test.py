import pandas as pd
import joblib

# Carregar modelo treinado
modelo = joblib.load('modelo_sieve_interval.pkl')

# Carregar novos dados
df_teste = pd.read_csv('teste.csv')
X_teste = df_teste[['glicose', 'pressao', 'temperatura']].values

# Previsão
y_pred = modelo.predict(X_teste)

# (Opcional) Avaliação se houver rótulos reais
from sklearn.metrics import classification_report
if 'classe_verdadeira' in df_teste.columns:
    print(classification_report(df_teste['classe_verdadeira'], y_pred))
