import pandas as pd
import joblib
import numpy as np


# Gerando novo CSV de peças para teste
np.random.seed(99)

n_teste = 10
dados_teste = {
    'comprimento': np.random.normal(101, 3, n_teste),
    'largura': np.random.normal(50, 1, n_teste),
    'peso': np.random.normal(200, 5, n_teste)
}
df_teste = pd.DataFrame(dados_teste)
df_teste.to_csv('pecas_teste.csv', index=False)

# Carregando e testando o modelo
modelo = joblib.load('modelo_sieve_pecas.pkl')
df_novos = pd.read_csv('pecas_teste.csv')
X_novos = df_novos[['comprimento', 'largura', 'peso']].values
y_pred = modelo.predict(X_novos)

classes_nomeadas = ['Reprovada', 'Retrabalho', 'Aprovada']
df_novos['classificacao_predita'] = [classes_nomeadas[i] for i in y_pred]
print(df_novos)

from sklearn.metrics import classification_report
print(classification_report(df_novos['classe'], y_pred, target_names=classes_nomeadas))
