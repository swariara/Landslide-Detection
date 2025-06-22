import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def train_stacked_model(X_train_scaled, X_val_scaled, X_test_scaled,
                        train_preds_cnn, val_preds_cnn, test_preds_cnn,
                        y_train, y_val,
                        xgb_model, lgb_model,
                        save_path):

    X_train_stack = np.hstack([X_train_scaled, train_preds_cnn.reshape(-1, 1)])
    X_val_stack = np.hstack([X_val_scaled, val_preds_cnn.reshape(-1, 1)])
    X_test_stack = np.hstack([X_test_scaled, test_preds_cnn.reshape(-1, 1)])

    stack_clf = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1,
        passthrough=True
    )

    stack_clf.fit(X_train_stack, y_train)
    y_pred_stack = stack_clf.predict(X_val_stack)

    print("Stacked F1:", f1_score(y_val, y_pred_stack))
    print(classification_report(y_val, y_pred_stack))

    cm = confusion_matrix(y_val, y_pred_stack)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['No Landslide', 'Landslide'], yticklabels=['No Landslide', 'Landslide'])
    plt.title("Confusion Matrix - Stacked Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path.parent / "confusion_matrix_stack.png")
    plt.close()

    joblib.dump(stack_clf, save_path)

    return X_train_stack, X_val_stack, X_test_stack

def predict_and_save_submissions(test_ids, test_preds_cnn_proba, test_preds_stack, cnn_out_path, stack_out_path):
    import pandas as pd
    pd.DataFrame({
        'id': test_ids,
        'label': (test_preds_cnn_proba >= 0.5).astype(int)
    }).to_csv(cnn_out_path, index=False)

    pd.DataFrame({
        'id': test_ids,
        'label': test_preds_stack
    }).to_csv(stack_out_path, index=False)
