# gda_train_tf.py
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Dataset load
data = pd.read_csv("disgenet_train.csv")

# Encode Gene_ID and Disease_ID
gene_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

data["Gene_ID_enc"] = gene_encoder.fit_transform(data["Gene_ID"])
data["Disease_ID_enc"] = disease_encoder.fit_transform(data["Disease_ID"])

# Train-test split
X = data[["Gene_ID_enc", "Disease_ID_enc"]]
y = data["association_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architecture (Embedding)
num_genes = len(gene_encoder.classes_)
num_diseases = len(disease_encoder.classes_)
embedding_dim = 32  # small dimension, can tune

gene_input = tf.keras.Input(shape=(1,), name="gene")
disease_input = tf.keras.Input(shape=(1,), name="disease")

gene_emb = tf.keras.layers.Embedding(input_dim=num_genes, output_dim=embedding_dim)(gene_input)
gene_emb = tf.keras.layers.Flatten()(gene_emb)

disease_emb = tf.keras.layers.Embedding(input_dim=num_diseases, output_dim=embedding_dim)(disease_input)
disease_emb = tf.keras.layers.Flatten()(disease_emb)

concat = tf.keras.layers.Concatenate()([gene_emb, disease_emb])
dense1 = tf.keras.layers.Dense(64, activation="relu")(concat)
dense2 = tf.keras.layers.Dense(32, activation="relu")(dense1)
output = tf.keras.layers.Dense(1)(dense2)

model = tf.keras.Model(inputs=[gene_input, disease_input], outputs=output)
model.compile(optimizer="adam", loss="mse")

# Train
history = model.fit(
    [X_train["Gene_ID_enc"], X_train["Disease_ID_enc"]],
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

# Predict
y_pred = model.predict([X_test["Gene_ID_enc"], X_test["Disease_ID_enc"]]).flatten()

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("===== Deep Learning Model =====")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Save predictions
pred_df = X_test.copy()
pred_df["True_Score"] = y_test
pred_df["Predicted_Score"] = y_pred
pred_df.to_csv("gda_test_predictions_dl.csv", index=False)

# Save model
model.save("gda_tf_model")
