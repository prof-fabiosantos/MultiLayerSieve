import numpy as np
import pandas as pd

np.random.seed(42)
n = 60

# Três classes: Reprovada (0), Retrabalho (1), Aprovada (2)
comprimento = np.concatenate([
    np.random.normal(97, 1.5, n//3),   # Reprovada (peças pequenas)
    np.random.normal(101, 1.0, n//3),  # Retrabalho (fora do ideal)
    np.random.normal(105, 1.5, n//3)   # Aprovada
])
largura = np.concatenate([
    np.random.normal(49, 1.2, n//3),
    np.random.normal(50, 0.7, n//3),
    np.random.normal(51, 0.9, n//3)
])
peso = np.concatenate([
    np.random.normal(195, 4, n//3),
    np.random.normal(200, 3, n//3),
    np.random.normal(205, 4, n//3)
])
classe = np.array([0]*(n//3) + [1]*(n//3) + [2]*(n//3))

df = pd.DataFrame({
    'comprimento': np.round(comprimento, 2),
    'largura': np.round(largura, 2),
    'peso': np.round(peso, 1),
    'classe': classe
})

df.to_csv('pecas_treino.csv', index=False)
print("Arquivo 'pecas_treino.csv' gerado!")
