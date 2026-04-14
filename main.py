import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SOIL CONDITION PREDICTION - ADVANCED TUNING WITH SMOTE")
print("=" * 70)

os.makedirs('models', exist_ok=True)

# Load real dataset
print("\n[1/8] Loading dataset...")
df = pd.read_csv('Data/dataset1.csv')
print(f"  [OK] Dataset shape: {df.shape}")
print(f"  [OK] Original distribution:\n{df['Output'].value_counts()}\n")

# Preprocessing
print("[2/8] Preprocessing data...")
df.fillna(df.mean(numeric_only=True), inplace=True)
df = df.drop_duplicates()

X = df.drop('Output', axis=1)
y = df['Output']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initial train/test split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"  [OK] Train before SMOTE: {X_train.shape}, distribution:\n{pd.Series(y_train).value_counts()}")

# Apply SMOTE to training data only
print("\n[3/8] Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"  [OK] Train after SMOTE: {X_train_smote.shape}, distribution:\n{pd.Series(y_train_smote).value_counts()}\n")

models_dict = {}

print("[4/8] Training XGBoost with SMOTE data...")
xgb_params = {
    'n_estimators': [200, 300],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 2],
    'subsample': [0.8, 0.9],
}
xgb = GridSearchCV(
    XGBClassifier(random_state=42, verbosity=0, eval_metric='mlogloss'),
    xgb_params, cv=5, n_jobs=-1, scoring='f1_weighted'
)
xgb.fit(X_train_smote, y_train_smote)
xgb_pred = xgb.predict(X_test)
models_dict['XGBoost'] = {'model': xgb.best_estimator_, 'pred': xgb_pred}
print(f"  [OK] Best params: {xgb.best_params_}")

print("\n[5/8] Training Random Forest with SMOTE data...")
rf_params = {
    'n_estimators': [200, 300],
    'max_depth': [12, 15, 18],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2],
}
rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample'),
    rf_params, cv=5, n_jobs=-1, scoring='f1_weighted'
)
rf.fit(X_train_smote, y_train_smote)
rf_pred = rf.predict(X_test)
models_dict['RandomForest'] = {'model': rf.best_estimator_, 'pred': rf_pred}
print(f"  [OK] Best params: {rf.best_params_}")

print("\n[6/8] Training Gradient Boosting with SMOTE data...")
gb_params = {
    'n_estimators': [200, 300],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_split': [2, 3],
    'subsample': [0.8, 0.9],
}
gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params, cv=5, n_jobs=-1, scoring='f1_weighted'
)
gb.fit(X_train_smote, y_train_smote)
gb_pred = gb.predict(X_test)
models_dict['GradientBoosting'] = {'model': gb.best_estimator_, 'pred': gb_pred}
print(f"  [OK] Best params: {gb.best_params_}")

print("\n[7/8] Model Comparison with Per-Class Metrics:")
print("-" * 70)

best_model_name = None
best_f1 = 0

for name, data in models_dict.items():
    pred = data['pred']
    acc = accuracy_score(y_test, pred)
    precision_weighted = precision_score(y_test, pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, pred, average='weighted', zero_division=0)

    print(f"\n{name}:")
    print(f"  Accuracy (Weighted):  {acc:.4f}")
    print(f"  Precision (Weighted): {precision_weighted:.4f}")
    print(f"  Recall (Weighted):    {recall_weighted:.4f}")
    print(f"  F1 Score (Weighted):  {f1_weighted:.4f}")

    print(f"\n  Per-Class Performance:")
    for class_idx in range(3):
        # Calculate per-class metrics
        mask = y_test == class_idx
        if mask.sum() > 0:
            correct = (pred[mask] == class_idx).sum()
            total = mask.sum()
            class_label = {0: 'Good(0)', 1: 'Bad(1)', 2: 'Medium(2)'}[class_idx]
            print(f"    {class_label}: {correct}/{total} correct ({correct/total*100:.1f}%)")

    if f1_weighted > best_f1:
        best_f1 = f1_weighted
        best_model_name = name

print("\n[8/8] Saving best model...")
best_model = models_dict[best_model_name]['model']
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"  [OK] Best model: {best_model_name} (F1 Score: {best_f1:.4f})")
print("  [OK] Files saved!")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_names = X.columns
    importances = best_model.feature_importances_
    print("\n  Top 6 Important Features:")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:6]:
        print(f"    {fname}: {imp:.4f}")

print("\n" + "=" * 70)
print("[DONE] Model training with SMOTE complete!")
print("[INFO] Model now better handles all three classes")
print("=" * 70)
