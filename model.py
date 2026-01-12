# File used to train XGBoost model for 4-hour A&E breach prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
 
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data\\AED4weeks.csv")

# =========================
# 0) Create/Clean Target Variable
# =========================

# Drop rows where 'Breachornot' is missing
df = df.dropna(subset=["Breachornot"]).copy()

# Convert target to binary: 1 if "breach", 0 otherwise
df["y_breach"] = (df["Breachornot"].astype(str).str.strip().str.lower() == "breach").astype(int)
 
# LoS (Length of Stay) is directly linked to the 4-hour breach definition. 
# It is removed to prevent data leakage.
drop_cols = [c for c in ["y_breach", "Breachornot", "LoS"] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df["y_breach"].astype(int)
 
print("Overall breach rate:", y.mean())
 
# =========================
# 1) Train/Test Split (20% test)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
 
print("\nTrain breach rate:", y_train.mean())
print("Test breach rate :", y_test.mean())
 
# =========================
# 2) Preprocessing (Numerical/Categorical)
# =========================
num_candidates = ["Age", "Period", "noofinvestigation", "nooftreatment", "noofpatients"]
cat_candidates = ["DayofWeek", "HRG", "Day"]
 
num_cols = [c for c in num_candidates if c in X.columns]
cat_cols = [c for c in cat_candidates if c in X.columns]
 
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)
 
# =========================
# 3) Handle Class Imbalance
# =========================
# Calculate scale_pos_weight to balance the classes
pos = int(y_train.sum())
neg = int(len(y_train) - pos)
scale_pos_weight = (neg / pos) if pos > 0 else 1.0
print(f"\nTrain positives={pos}, negatives={neg}, scale_pos_weight={scale_pos_weight:.2f}")
 
# =========================
# 4) XGBoost Model Training
# =========================
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
 
model = Pipeline(steps=[("prep", preprocess), ("clf", xgb)])
model.fit(X_train, y_train)
 
# ==============================================================
# 5) Predict Probabilities + Threshold Optimization (Youden's J)
# ==============================================================
proba_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)
 
fpr, tpr, thr = roc_curve(y_test, proba_test)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_thr = thr[best_idx]
 
print("\nROC-AUC:", round(auc, 3))
print("ROC-optimal threshold (Youden J):", float(best_thr))
 
# ==================================================
# 6) Classification with Optimal Threshold + Results
# ==================================================
pred_test = (proba_test >= best_thr).astype(int)
 
print("\nPredicted breaches:", int(pred_test.sum()))
print("Actual breaches   :", int(y_test.sum()))
 
cm = confusion_matrix(y_test, pred_test)
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, pred_test, digits=3))
 
# =========================
# 7) Visualization: ROC + Confusion Matrix
# =========================
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.2f})")
plt.scatter(fpr[best_idx], tpr[best_idx], label=f"Best thr={best_thr:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – XGBoost (test 20%)")
plt.legend()
plt.tight_layout()
plt.show()
 
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix – XGBoost (ROC-optimal threshold)")
plt.colorbar()
plt.xticks([0,1], ["Pred Non-breach", "Pred Breach"], rotation=15)
plt.yticks([0,1], ["True Non-breach", "True Breach"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# Save the trained model
import joblib
from joblib import dump
dump(model, "xgb_model.pkl")