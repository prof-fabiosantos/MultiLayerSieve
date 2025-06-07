# MultiLayer Sieve

> Um classificador supervisionado inspirado em peneiras multicamadas, com thresholds aprendíveis e alta interpretabilidade.

---

## Visão Geral

O **MultiLayer Sieve** é um algoritmo de classificação supervisionada inspirado no funcionamento físico das peneiras empilhadas, muito comuns em processos industriais e laboratoriais.  
Cada camada da peneira corresponde a um conjunto de thresholds (malhas) — amostras são “triadas” camada por camada, de acordo com regras baseadas em valores numéricos simples.  
Os thresholds de cada camada são **ajustados automaticamente** para maximizar a performance nos dados de treino.

---

## Lógica do Algoritmo

- Cada camada da peneira tem um threshold para cada feature numérica.
- Uma amostra é classificada em uma camada se todas as features forem menores que os thresholds daquela camada.
- Amostras não classificadas seguem para a próxima camada; quem não é classificado em nenhuma recebe a última classe.
- Os thresholds são aprendidos via busca em grade (`GridSearchCV`), garantindo flexibilidade e performance.
- O modelo pode ser salvo/carregado (persistência total) e é integrado ao estilo scikit-learn.

---

## Vantagens

- **Explicável:** Decisões são 100% rastreáveis aos thresholds de cada camada.
- **Flexível:** Número de camadas, features e granularidade dos thresholds são facilmente ajustáveis.
- **Reutilizável:** Compatível com pipelines scikit-learn e persistência via joblib.
- **Aplicável em contextos de triagem, inspeção e separação de grupos bem definidos por faixas numéricas.**

---

## Exemplos de Aplicação

- **Inspeção industrial:** Classificação de peças ou produtos por características medidas.
- **Pré-triagem médica:** Separação de pacientes em níveis de urgência com base em exames simples.
- **Análise de crédito:** Segmentação por faixas de renda, score, idade, etc.
- **Agronegócio:** Classificação de alimentos por peso, tamanho ou teor de umidade.

---

## Limitações

- Não recomendado para relações altamente não-lineares.
- GridSearch pode ser custoso para muitos features/camadas/thresholds.
- Melhor desempenho em problemas interpretáveis, datasets pequenos/médios ou prototipagem rápida.

## Instalação

Basta copiar o arquivo `learnable_sieve.py` para o seu projeto Python.  
É necessário ter `numpy`, `pandas` e `scikit-learn` instalados.

```bash
pip install numpy pandas scikit-learn joblib
