import numpy as np
import pandas as pd

np.random.seed(100)
n_amostras = 60

glicose = np.concatenate([
    np.random.normal(62, 2, n_amostras//3),   # Urgência alta
    np.random.normal(75, 2, n_amostras//3),   # Urgência moderada
    np.random.normal(95, 8, n_amostras//3)    # Urgência baixa
])

pressao = np.concatenate([
    np.random.normal(172, 3, n_amostras//3),  # Urgência alta
    np.random.normal(135, 3, n_amostras//3),  # Urgência moderada
    np.random.normal(119, 5, n_amostras//3)   # Urgência baixa
])

temperatura = np.concatenate([
    np.random.normal(38.6, 0.2, n_amostras//3),  # Urgência alta
    np.random.normal(37.1, 0.2, n_amostras//3),  # Urgência moderada
    np.random.normal(36.5, 0.2, n_amostras//3)   # Urgência baixa
])

# Classe verdadeira baseada na origem dos dados
classe_verdadeira = np.array([0]*(n_amostras//3) + [1]*(n_amostras//3) + [2]*(n_amostras//3))

df_teste = pd.DataFrame({
    'glicose': np.round(glicose, 1),
    'pressao': np.round(pressao, 1),
    'temperatura': np.round(temperatura, 1),
    'classe_verdadeira': classe_verdadeira
})

df_teste.to_csv('novos_pacientes.csv', index=False)
print("Arquivo 'novos_pacientes.csv' gerado com 60 exemplos balanceados!")
