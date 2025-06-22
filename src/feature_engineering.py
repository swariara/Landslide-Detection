import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

def calculate_band_stats(img):
    mean = img.mean(axis=(0, 1))
    std = img.std(axis=(0, 1))
    mn = img.min(axis=(0, 1))
    mx = img.max(axis=(0, 1))
    return np.concatenate([mean, std, mn, mx])

def calculate_ndvi(img):
    nir = img[:, :, 3]
    red = img[:, :, 0]
    ndvi = (nir - red) / (nir + red + 1e-5)
    return np.array([ndvi.mean(), ndvi.std()])

def calculate_band_ratios(img):
    nir = img[:, :, 3]
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    return np.array([
        np.mean(nir / (red + 1e-5)),
        np.mean(nir / (green + 1e-5)),
        np.mean(nir / (blue + 1e-5))
    ])

def prepare_features(df, folder_path):
    features = []
    for image_id in df['id'].values:
        img = np.load(folder_path / f"{image_id}.npy")
        stats = calculate_band_stats(img)
        ndvi = calculate_ndvi(img)
        ratios = calculate_band_ratios(img)
        combined = np.concatenate([stats, ndvi, ratios])
        features.append(combined)
    return np.array(features)

def create_scaler(X_train, X_val, X_test, save_path):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, save_path)
    return {"train": X_train_scaled, "val": X_val_scaled, "test": X_test_scaled}
