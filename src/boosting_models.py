from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

def train_xgboost(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    clf = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf.best_estimator_

def train_lightgbm(X_train, y_train):
    param_grid = {
        'num_leaves': [31, 50],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    clf = GridSearchCV(
        LGBMClassifier(),
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf.best_estimator_
