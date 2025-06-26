# 🏔️ Landslide Detection Using Multi-Band Satellite Imagery

## 📌 Project Overview

This project builds a robust machine learning pipeline for detecting landslides using satellite imagery with 12 spectral bands, including both optical and radar data. The goal is to classify each image as either **Landslide** or **No Landslide**.

To achieve high accuracy, the pipeline integrates:

- A custom Convolutional Neural Network (CNN) built with Keras and tuned using Keras Tuner  
- Feature engineering to extract statistical and spectral features from image bands  
- Boosting models using XGBoost and LightGBM  
- A final stacking ensemble that combines deep learning and boosting predictions for improved performance  

---

## 🧰 Tech Stack

**Languages & Libraries**  
Python, NumPy, Pandas, Matplotlib, Seaborn  

**Deep Learning**  
TensorFlow, Keras, Keras-Tuner  

**Machine Learning**  
scikit-learn, XGBoost, LightGBM, scikeras  

**Utilities**  
joblib, Google Colab, Google Drive

---
## 🚀 Approach

### 1. 📥 Data Preprocessing

- Cleaned and normalized image bands  
- Stratified train/validation split (80/20)  
- Computed class weights to address label imbalance

### 2. 🧠 Model 1: CNN (Keras)

- Trained on 12-band satellite images  
- Custom data generator with augmentation  
- Keras Tuner for hyperparameter search  
- F1 Score-based model checkpointing

### 3. 🔬 Feature Engineering

- Band statistics: mean, std, min, max  
- NDVI calculation  
- Band ratios: NIR/Red, NIR/Green, NIR/Blue  

### 4. 🌲 Model 2: XGBoost & LightGBM

- Used statistical + spectral features  
- Grid search for hyperparameter tuning  
- Evaluated on validation F1 Score  

### 5. 🧬 Model 3: Stacked Ensemble

- Base learners: XGBoost, LightGBM  
- Meta learner: Logistic Regression  
- CNN predictions used as additional feature  
- Final model trained using StackingClassifier

---

## 📊 Evaluation

Models were evaluated on the validation set using:

- F1 Score  
- Precision, Recall  
- Confusion Matrix  

| Model       | Validation F1 Score |
|-------------|---------------------|
| CNN         | ✅ Reported         |
| XGBoost     | ✅ Reported         |
| LightGBM    | ✅ Reported         |
| Stacked     | ✅ Best Performance |

---

## 🛠️ How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/landslide-detection.git
cd landslide-detection

# Install dependencies
pip install keras-tuner scikeras xgboost lightgbm

# Upload Train.csv, Test.csv, train_data/, test_data/ to your environment

# Run the training pipeline (via notebook or Python script)



