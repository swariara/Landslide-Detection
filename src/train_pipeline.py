import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report
import joblib
import keras_tuner as kt
import tensorflow as tf
from matplotlib import pyplot as plt

from utils.data_generator import LandslideDataGenerator
from src.cnn_model import build_cnn_model, F1ScoreCallback
from src.feature_engineering import prepare_features, create_scaler
from src.boosting_models import train_xgboost, train_lightgbm
from src.ensemble import train_stacked_model, predict_and_save_submissions


# === Paths ===
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
MODEL_PATH = ROOT / "models"
TUNER_PATH = ROOT / "kt_tuner_dir"
OUTPUT_PATH = ROOT / "outputs"

for path in [MODEL_PATH, TUNER_PATH, OUTPUT_PATH]:
    path.mkdir(parents=True, exist_ok=True)


# === Load Data ===
train_df = pd.read_csv(DATA_PATH / "Train.csv")
test_df = pd.read_csv(DATA_PATH / "Test.csv")
train_df.columns = train_df.columns.str.strip().str.lower()
test_df.columns = test_df.columns.str.strip().str.lower()


# === Train/Val Split ===
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)
y_train = train_df['label'].values
y_val = val_df['label'].values


# === Class Weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))


# === Data Generators ===
train_gen = LandslideDataGenerator(train_df['id'].values, y_train, DATA_PATH / "train_data", augment=True)
val_gen = LandslideDataGenerator(val_df['id'].values, y_val, DATA_PATH / "train_data", shuffle=False)


# === CNN Model Tuning & Training ===
cnn_save_path = MODEL_PATH / "best_cnn_model.keras"
f1_callback = F1ScoreCallback(val_gen, save_path=cnn_save_path)

tuner = kt.RandomSearch(
    build_cnn_model,
    objective="val_accuracy",
    max_trials=3,
    directory=TUNER_PATH,
    project_name="landslide_cnn"
)

tuner.search(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        f1_callback
    ]
)

best_cnn = tf.keras.models.load_model(cnn_save_path)
val_preds_cnn = (best_cnn.predict(val_gen) >= 0.5).astype(int)
print("\n=== CNN Evaluation ===")
print("F1 Score:", f1_score(y_val, val_preds_cnn))
print(classification_report(y_val, val_preds_cnn))


# === Feature Engineering ===
X_train = prepare_features(train_df, DATA_PATH / "train_data")
X_val = prepare_features(val_df, DATA_PATH / "train_data")
X_test = prepare_features(test_df, DATA_PATH / "test_data")

scaler_outputs = create_scaler(X_train, X_val, X_test, MODEL_PATH / "scaler.pkl")
X_train_scaled = scaler_outputs["train"]
X_val_scaled = scaler_outputs["val"]
X_test_scaled = scaler_outputs["test"]


# === Boosting Models ===
print("\n=== Training XGBoost ===")
xgb_model = train_xgboost(X_train_scaled, y_train)

print("\n=== Training LightGBM ===")
lgb_model = train_lightgbm(X_train_scaled, y_train)


# === CNN Probabilities for Stacking ===
train_gen_noshuffle = LandslideDataGenerator(train_df['id'].values, y_train, DATA_PATH / "train_data", shuffle=False)
val_preds_cnn_proba = best_cnn.predict(val_gen).flatten()
train_preds_cnn_proba = best_cnn.predict(train_gen_noshuffle).flatten()

test_gen = LandslideDataGenerator(test_df['id'].values, [0]*len(test_df), DATA_PATH / "test_data", shuffle=False)
test_preds_cnn_proba = best_cnn.predict(test_gen).flatten()


# === Stacking Ensemble ===
print("\n=== Training Stacking Model ===")
train_stacked, val_stacked, test_stacked = train_stacked_model(
    X_train_scaled, X_val_scaled, X_test_scaled,
    train_preds_cnn_proba, val_preds_cnn_proba, test_preds_cnn_proba,
    y_train, y_val,
    xgb_model, lgb_model,
    MODEL_PATH / "stacked_model.pkl"
)


# === Save Predictions ===
print("\n=== Saving Submissions ===")
predict_and_save_submissions(
    test_df['id'].values,
    test_preds_cnn_proba,
    test_stacked,
    OUTPUT_PATH / "cnn_submission.csv",
    OUTPUT_PATH / "stacked_submission.csv"
)


# === Save Models ===
print("\n=== Saving Models ===")
joblib.dump(xgb_model, MODEL_PATH / "best_xgb_model.pkl")
joblib.dump(lgb_model, MODEL_PATH / "best_lgbm_model.pkl")
