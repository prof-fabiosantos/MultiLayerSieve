import numpy as np
import pandas as pd

np.random.seed(42)
n_amostras = 60  # 20 para cada classe

# Urgência alta
glicose_alta = np.random.normal(62, 3, n_amostras//3)      # Baixa glicose
pressao_alta = np.random.normal(172, 5, n_amostras//3)     # Pressão alta
temperatura_alta = np.random.normal(38.7, 0.2, n_amostras//3) # Temperatura alta

# Urgência moderada
glicose_moderada = np.random.normal(75, 3, n_amostras//3)
pressao_moderada = np.random.normal(138, 5, n_amostras//3)
temperatura_moderada = np.random.normal(37.2, 0.2, n_amostras//3)

# Urgência baixa
glicose_baixa = np.random.normal(95, 7, n_amostras//3)
pressao_baixa = np.random.normal(120, 7, n_amostras//3)
temperatura_baixa = np.random.normal(36.6, 0.2, n_amostras//3)

# Juntando tudo
glicose = np.concatenate([glicose_alta, glicose_moderada, glicose_baixa])
pressao = np.concatenate([pressao_alta, pressao_moderada, pressao_baixa])
temperatura = np.concatenate([temperatura_alta, temperatura_moderada, temperatura_baixa])
classe_verdadeira = np.array([0]*(n_amostras//3) + [1]*(n_amostras//3) + [2]*(n_amostras//3))

df = pd.DataFrame({
    'glicose': np.round(glicose, 1),
    'pressao': np.round(pressao, 1),
    'temperatura': np.round(temperatura, 1),
    'classe_verdadeira': classe_verdadeira
})

df.to_csv('treino.csv', index=False)
print("Arquivo treino.csv criado com 60 exemplos balanceados!")
