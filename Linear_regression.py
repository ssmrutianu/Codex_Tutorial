"""
multi_output_linreg.py
Train a Linear Regression model with 1 feature (X)
and 2 target variables (y1, y2).  Reports MSE, MAE, and R²
for each output. Add new features.
"""
 
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
 
# ───────────────────────────── 1. Make a sample data-set
# 120 samples, 1 input feature, 2 output targets
X, y = make_regression(
    n_samples=120,
    n_features=1,
    n_targets=2,      # <-- two dependent variables
    noise=10,
    random_state=42,
)
 
# ───────────────────────────── 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# ───────────────────────────── 3. Fit multi-output regression
model = LinearRegression()
model.fit(X_train, y_train)
 
# ───────────────────────────── 4. Predict & evaluate
y_pred = model.predict(X_test)
 
# scikit-learn metrics accept multi-output arrays; set multioutput='raw_values'
mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
r2  = r2_score(y_test, y_pred, multioutput="raw_values")
 
target_names = ["y₁", "y₂"]
print("\n=== Baseline metrics per target ===")
for i, t in enumerate(target_names):
    print(f"{t}:  MSE={mse[i]:.2f}  MAE={mae[i]:.2f}  R²={r2[i]:.3f}")