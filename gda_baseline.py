# gda_baseline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dataset load
data = pd.read_csv("disgenet_train.csv")

# Train-test split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Baseline: প্রতিটা prediction = training set এর গড় score
mean_score = train["Score"].mean()
test["baseline_pred"] = mean_score

# Evaluation
mse = mean_squared_error(test["Score"], test["baseline_pred"])
mae = mean_absolute_error(test["Score"], test["baseline_pred"])
r2 = r2_score(test["Score"], test["baseline_pred"])

print("===== Baseline Model =====")
print(f"Mean Score Used: {mean_score:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Save predictions
test.to_csv("gda_test_predictions_baseline.csv", index=False)
