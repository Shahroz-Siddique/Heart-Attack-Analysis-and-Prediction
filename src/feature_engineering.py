import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def select_k_best_features(X, y, k=8):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=selected_features), selected_features
