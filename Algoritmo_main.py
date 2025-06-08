def peneira_multicamadas(dados, thresholds):
    classes = []
    for valor in dados:
        classificado = False
        for i, limiar in enumerate(thresholds):
            if valor < limiar:
                classes.append(i)
                classificado = True
                break
        if not classificado:
            classes.append(len(thresholds))
    return classes

dados = [3, 8, 12, 7, 2, 15]
thresholds = [5, 10, 13]  # Camada 1: <5, Camada 2: <10, Camada 3: <13
print(peneira_multicamadas(dados, thresholds))
# Resultado: [0, 1, 3, 1, 0, 3]
