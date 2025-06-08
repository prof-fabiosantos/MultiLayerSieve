# MultiLayer Sieve

O **MultiLayer Sieve** é um algoritmo de classificação supervisionada inspirado no funcionamento físico das peneiras dos povos originários da Amazônia.

<p align="center">
  <img src="./logo.png" alt="Logo Transformer Core" width="300"/>
</p>
 
---

## Algoritmo

O **MultiLayer Sieve** é um algoritmo de classificação supervisionada inspirado no funcionamento físico das peneiras empilhadas. Cada camada corresponde a um conjunto de thresholds (malhas) — amostras são “triadas” camada por camada, de acordo com regras baseadas em valores numéricos simples. Os thresholds são ajustados automaticamente para maximizar a performance nos dados de treino.

---

## Formalização

Cada amostra \$x \in \mathbb{R}^d\$ passa sequencialmente por camadas (peneiras) \$S\_k\$, onde cada camada aplica um teste de thresholds por feature:

$$
S_k(x) =
\begin{cases}
  c_k & \text{se } x_j < t_{k, j} \ \forall j \in [1, d] \\
  \text{segue para próxima camada} & \text{caso contrário}
\end{cases}
$$

* \$c\_k\$: classe atribuída na camada \$k\$
* \$t\_{k, j}\$: threshold para a feature \$j\$ na camada \$k\$

**Com intervalos (versão avançada):**

$$
S_k(x) =
\begin{cases}
  c_k & \text{se } t_{k,j}^{min} < x_j < t_{k,j}^{max} \ \forall j \in [1, d] \\
  \text{segue para próxima camada} & \text{caso contrário}
\end{cases}
$$

As amostras são processadas em sequência:

$$
x \xrightarrow{S_1} y_1 \xrightarrow{S_2} y_2 \dots \xrightarrow{S_K} \text{classe final}
$$

Em cada camada, se a condição for satisfeita, a classe \$c\_k\$ é atribuída e a triagem termina.
Se não for satisfeita em nenhuma camada, a amostra recebe uma classe padrão (ex: "aprovado" ou última classe).

### Pseudocódigo

```python
# Percorre todas as camadas/peneiras do modelo (K é o número de camadas). Ex: Se K = 3, ele verifica a camada 0, depois a 1, depois a 2.
for k in range(K):
    # Para a camada k, verifica todas as features (variáveis de entrada) da amostra x:
    # x[j] < t_kj significa: O valor da feature j está abaixo do threshold da camada k para aquela feature.
    # all(...) exige que essa condição seja verdadeira para todas as features da amostra.
    # Versão com intervalos:
    # tmin_kj < x[j] < tmax_kj — o valor da feature precisa cair dentro do intervalo permitido para aquela camada e feature (não só menor que o threshold, mas entre dois limites).
    if all(x[j] < t_kj for j in range(d)): # ou: if all(tmin_kj < x[j] < tmax_kj)  
        # Se a amostra x passou pelo filtro da camada k, então ela é classificada com a classe associada àquela camada (c_k), e a execução termina aqui para essa amostra.
        return c_k 
# Se não passou em nenhuma peneira
return classe_padrao
```

---

## Justificativa e Aplicações

Embora exista uma grande variedade de algoritmos supervisionados para problemas lineares, como Regressão Logística, SVM linear, LDA e outros, o **MultiLayer Sieve** se destaca por sua proposta centrada em interpretabilidade, transparência e aderência a processos de decisão já praticados em ambientes industriais, médicos e corporativos.

### Por que propor o MultiLayer Sieve?

* **Interpretabilidade Total:**
  O MultiLayer Sieve transforma o processo de classificação em um conjunto claro de regras em cascata. Cada decisão pode ser explicada diretamente com base nos thresholds de cada camada, tornando o modelo auditável e compreensível por profissionais de qualquer área.

* **Aderência a Processos Reais:**
  Muitos fluxos de triagem e inspeção já utilizam regras de faixas e etapas sequenciais para tomada de decisão (ex: rejeição imediata, retrabalho, aprovação). O MultiLayer Sieve reflete e digitaliza fielmente esse raciocínio, facilitando aceitação e implantação.

* **Customização e Modularidade:**
  O algoritmo permite fácil personalização: número de camadas, ranges de thresholds, lógicas de decisão em cada etapa, e até a integração de modelos diferentes em cada camada.

* **Auditabilidade e Regulamentação:**
  Em setores regulados (saúde, indústria, finanças), a rastreabilidade da decisão é fundamental. O MultiLayer Sieve fornece explicações claras para cada classificação, atendendo requisitos de compliance e auditoria.

* **Simplicidade e Educação:**
  Por ser conceitualmente simples e visual, o MultiLayer Sieve é excelente para ensino, prototipagem e validação inicial de hipóteses, além de facilitar a colaboração entre cientistas de dados e especialistas do domínio.

### Possíveis Aplicações

* **Inspeção e Controle de Qualidade Industrial:**
  Classificação de peças/produtos conforme faixas de medidas, testes físicos ou padrões normativos.

* **Triagem Médica ou Laboratorial:**
  Priorização de casos, alertas de urgência e separação de grupos de pacientes com base em exames e sinais clínicos.

* **Análise e Concessão de Crédito:**
  Segmentação de propostas em categorias de risco segundo critérios objetivos (faixas de renda, idade, score, etc.).

* **Agronegócio:**
  Classificação de produtos agrícolas conforme padrões de tamanho, peso, umidade, pureza, etc.

* **Qualquer contexto onde regras claras por faixas e etapas sejam valorizadas ou exigidas.**

### Conclusão

O MultiLayer Sieve não busca substituir métodos lineares clássicos, mas sim **preencher um nicho de aplicabilidade onde a explicação, a aderência ao processo e a transparência são mais importantes que a complexidade matemática ou a última fração de acurácia**.
Sua principal força reside na facilidade de auditoria, adaptação a processos já existentes e ganho de confiança por parte dos usuários finais.

---

## Instalação

Basta copiar o arquivo `learnable_sieve.py` para o seu projeto Python.
É necessário ter `numpy`, `pandas` e `scikit-learn` instalados.

```bash
pip install numpy pandas scikit-learn joblib
```

---

## Exemplo de Uso

### Treinamento

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV
from learnable_sieve import LearnableMultiLayerSieve
import joblib

df = pd.read_csv('treino.csv')
X = df[['glicose', 'pressao', 'temperatura']].values
y = df['classe_verdadeira'].values

clf = MultiLayerSieve(
    n_layers=2,
    feature_names=['glicose', 'pressao', 'temperatura'],
    n_thresholds=10
)
clf.fit(X, y)                # Gera os ranges automaticamente!
param_grid = clf.get_param_grid()
grid = GridSearchCV(clf, param_grid, scoring='accuracy', cv=2)
grid.fit(X, y)

joblib.dump(grid.best_estimator_, 'modelo_sieve.pkl')
```

### Teste e Avaliação

```python
import pandas as pd
import joblib

modelo = joblib.load('modelo_sieve.pkl')
df_teste = pd.read_csv('teste.csv')
X_teste = df_teste[['glicose', 'pressao', 'temperatura']].values
y_pred = modelo.predict(X_teste)

from sklearn.metrics import classification_report, confusion_matrix
if 'classe_verdadeira' in df_teste.columns:
    print(classification_report(df_teste['classe_verdadeira'], y_pred))
```

---

## Hiperparâmetros do MultiLayer Sieve

| Hiperparâmetro      | Descrição / O que faz                                                                                                                                                                                                                                                                                                            | Quando usar / Dica                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **n\_layers**       | Número de camadas (“peneiras”) empilhadas no modelo.<br><br>**O que faz:**<br>- Cada camada representa uma etapa de triagem, com thresholds próprios para cada feature.<br>- Quanto mais camadas, mais refinada pode ser a separação entre classes.                                                                              | **Quando aumentar/diminuir:**<br>- Use mais camadas se seu problema tiver múltiplas faixas de classificação (ex: rejeitado, retrabalho, aprovado).<br>- Use menos para casos binários ou mais simples. |
| **feature\_ranges** | Dicionário com o intervalo de valores a serem testados como thresholds para cada feature (`{feature_name: array_de_valores}`).<br><br>**O que faz:**<br>- Permite customizar a busca dos melhores thresholds para cada feature.                                                                                                  | **Quando usar:**<br>- Se quiser controlar os possíveis valores de thresholds (por exemplo, para priorizar faixas específicas ou acelerar a busca).                                                     |
| **n\_thresholds**   | Quantidade de valores a serem gerados automaticamente para cada feature ao criar os ranges de thresholds.<br><br>**O que faz:**<br>- Define a granularidade da busca:<br>  • Valores mais altos → thresholds mais precisos, porém busca mais lenta<br>  • Valores mais baixos → thresholds menos precisos, mas busca mais rápida | **Dica:**<br>- Ajuste conforme o tamanho do seu dataset e o poder computacional disponível.                                                                                                            |

---

## Estrutura dos arquivos

* `learnable_sieve.py` — Implementação do classificador MultiLayer Sieve.
* `TrainMultiLayerSieve.py` — Exemplo de script de treinamento.
* `TestMultiLayerSieve.py` — Exemplo de script de teste/avaliação.
* `treino.csv`, `teste.csv` — Dados de exemplo (colunas: features numéricas + classe).

---

## Limitações

* **Não recomendado para relações altamente não-lineares**.
* GridSearch pode ser custoso para muitos features/camadas/thresholds.
* Melhor desempenho em problemas interpretáveis, datasets pequenos/médios ou prototipagem rápida.

---

## Contribuindo

Sinta-se livre para propor melhorias, sugerir issues, ou criar pull requests para:

* Outras funções de decisão por camada (ex: operadores diferentes de `<`)
* Suporte a mais tipos de dados
* Integração com mais pipelines

---

## Licença

MIT

---

> Dúvidas, sugestões ou exemplos de uso? Abra uma issue ou entre em contato!


