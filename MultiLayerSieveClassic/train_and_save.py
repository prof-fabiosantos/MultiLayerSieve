import joblib
from sklearn.model_selection import GridSearchCV
from MultiLayerSieveClassic import MultiLayerSieveClassic

def fit_and_save_model(
    X, y,
    feature_names,
    n_layers=2,
    n_thresholds=5,
    scoring='accuracy',
    cv=2,
    model_path='modelo_sieve.pkl',
    verbose=2,
    n_jobs=-1
):
    clf = MultiLayerSieveClassic(
        n_layers=n_layers,
        feature_names=feature_names,
        n_thresholds=n_thresholds
    )
    # Passe X para o param_grid:
    param_grid = clf.get_param_grid(X)
    grid = GridSearchCV(
        clf,
        param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs
    )
    grid.fit(X, y)
    import joblib
    joblib.dump(grid.best_estimator_, model_path)
    print(f"Modelo salvo em {model_path}. Melhor score: {grid.best_score_:.3f}")
    return grid


