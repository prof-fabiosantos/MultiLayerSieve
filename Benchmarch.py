import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from MultiLayerSieveClassic import MultiLayerSieveClassic
from sklearn.model_selection import GridSearchCV

# Gerar dados: classe 0 se x < 0.3 e y < 0.7, senão classe 1
# Gerar dados: classe 1 se x + y > 1, senão classe 0
np.random.seed(2)
X = np.random.rand(300, 2)
y = ((X[:,0] + X[:,1]) > 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# MultiLayer Sieve
clf = MultiLayerSieveClassic(
    n_layers=2,
    feature_names=['f1', 'f2'],
    n_thresholds=5
)
clf.fit(X_train, y_train)
param_grid = clf.get_param_grid()
grid = GridSearchCV(clf, param_grid, cv=2)
grid.fit(X_train, y_train)
y_pred_sieve = grid.best_estimator_.predict(X_test)

# Logística
logreg = LogisticRegression().fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)

# SVM linear
svm = SVC(kernel='linear').fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("\nProblema Linear Oblíquo (x+y>1)")
print("Sieve:", accuracy_score(y_test, y_pred_sieve))
print("Logística:", accuracy_score(y_test, y_pred_log))
print("SVM Linear:", accuracy_score(y_test, y_pred_svm))
