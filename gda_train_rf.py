#!/usr/bin/env python3
# gda_train_rf.py
# Trains a RandomForest regression model on Gene-Disease association CSV
# Usage: put disgenet_train.csv in same folder and run: python gda_train_rf.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

INPUT = "disgenet_train.csv"  # adjust if needed
DF = pd.read_csv(INPUT)

# Columns (change these if your CSV uses different names)
GENE_COL = "Gene_ID"
DISEASE_COL = "Disease_ID"
SCORE_COL = "Y"

df = DF[[GENE_COL, DISEASE_COL, SCORE_COL]].dropna().reset_index(drop=True)
df[GENE_COL] = df[GENE_COL].astype(str)
df[DISEASE_COL] = df[DISEASE_COL].astype(str)

# Feature engineering: target encoding and counts
df["gene_mean"] = df.groupby(GENE_COL)[SCORE_COL].transform("mean")
df["disease_mean"] = df.groupby(DISEASE_COL)[SCORE_COL].transform("mean")
df["gene_count"] = df.groupby(GENE_COL)[SCORE_COL].transform("count")
df["disease_count"] = df.groupby(DISEASE_COL)[SCORE_COL].transform("count")
df["interaction"] = df["gene_mean"] * df["disease_mean"]

FEATURES = ["gene_mean", "disease_mean", "gene_count", "disease_count", "interaction"]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

X_train = train_df[FEATURES].values
y_train = train_df[SCORE_COL].values
X_test = test_df[FEATURES].values
y_test = test_df[SCORE_COL].values

print("Training RandomForest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"RF results -> MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")

# Save model and predictions
joblib.dump(rf, "gda_rf_model.joblib")
out = test_df[[GENE_COL, DISEASE_COL, SCORE_COL, "gene_mean", "disease_mean"]].copy()
out["pred_rf"] = preds
out.to_csv("gda_test_predictions_rf.csv", index=False)
print("Saved model -> gda_rf_model.joblib and predictions -> gda_test_predictions_rf.csv")