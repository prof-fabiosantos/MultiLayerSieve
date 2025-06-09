import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from MultiLayerSieveInterval import MultiLayerSieveInterval

# Carrega o dataset Iris
X, y = load_iris(return_X_y=True)
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Hiperparâmetros do MultiLayerSieveInterval
n_layers = 2
n_intervals = 4
scoring = 'accuracy'
cv = 3
verbose = 2
n_jobs = -1

# MultiLayerSieveInterval com GridSearchCV externo
clf = MultiLayerSieveInterval(
    n_layers=n_layers,
    feature_names=feature_names,
    n_intervals=n_intervals
)
param_grid = clf.get_param_grid(X_train)
grid = GridSearchCV(
    clf,
    param_grid,
    scoring=scoring,
    cv=cv,
    verbose=verbose,
    n_jobs=n_jobs
)
grid.fit(X_train, y_train)
y_pred_sieve = grid.predict(X_test)
acc_sieve = accuracy_score(y_test, y_pred_sieve)

# Algoritmos clássicos para comparação
results = {}

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
results['DecisionTree'] = accuracy_score(y_test, tree.predict(X_test))

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
results['LogisticRegression'] = accuracy_score(y_test, logreg.predict(X_test))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
results['KNN'] = accuracy_score(y_test, knn.predict(X_test))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
results['RandomForest'] = accuracy_score(y_test, rf.predict(X_test))

results['MultiLayerSieveInterval'] = acc_sieve

# Exibe resultados
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
