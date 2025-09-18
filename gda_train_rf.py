# gda_train_rf.py
# RandomForest model for Gene-Disease score prediction

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

data = pd.read_csv("disgenet_train.csv")


df = data[["Gene_ID", "Disease_ID", "Y"]].dropna().reset_index(drop=True)
df["Gene_ID"] = df["Gene_ID"].astype(str)
df["Disease_ID"] = df["Disease_ID"].astype(str)


df["g_mean"] = df.groupby("Gene_ID")["Y"].transform("mean")
df["d_mean"] = df.groupby("Disease_ID")["Y"].transform("mean")
df["g_cnt"] = df.groupby("Gene_ID")["Y"].transform("count")
df["d_cnt"] = df.groupby("Disease_ID")["Y"].transform("count")
df["mix"] = df["g_mean"] * df["d_mean"]

feat = ["g_mean", "d_mean", "g_cnt", "d_cnt", "mix"]

train, test = train_test_split(df, test_size=0.2, shuffle=True)

Xtr, ytr = train[feat].values, train["Y"].values
Xte, yte = test[feat].values, test["Y"].values

print("Fitting RandomForest...")
model = RandomForestRegressor(n_estimators=120, max_depth=12, n_jobs=-1)
model.fit(Xtr, ytr)

pred = model.predict(Xte)

print("MSE:", mean_squared_error(yte, pred))
print("MAE:", mean_absolute_error(yte, pred))
print("R2:", r2_score(yte, pred))

joblib.dump(model, "rf_model.pkl")

out = test[["Gene_ID", "Disease_ID", "Y"]].copy()
out["pred"] = pred
out.to_csv("rf_test_out.csv", index=False)
print("done, files saved")
