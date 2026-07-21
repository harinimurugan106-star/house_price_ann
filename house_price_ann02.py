import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("housing_price_dataset.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================
# Feature Selection
# ==========================
X = df.drop(["property_id", "price_in_lakhs", "price_category"], axis=1)

y = df["price_in_lakhs"]

# Convert categorical columns into numerical values
X = pd.get_dummies(X, drop_first=True)

# ==========================
# Split Dataset
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Feature Scaling
# ==========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Build Multilayer Feed Forward Neural Network
# ==========================
model = Sequential()

# Input Layer + Hidden Layer 1
model.add(Dense(64,
                activation='relu',
                input_shape=(X_train.shape[1],)))

model.add(Dropout(0.20))

# Hidden Layer 2
model.add(Dense(32,
                activation='relu'))

model.add(Dropout(0.20))

# Hidden Layer 3
model.add(Dense(16,
                activation='relu'))

# Output Layer
model.add(Dense(1,
                activation='linear'))

# ==========================
# Compile Model
# ==========================
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

print("\nModel Summary")
model.summary()

# ==========================
# Early Stopping
# ==========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ==========================
# Train Model
# ==========================
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)

# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Evaluation Metrics
# ==========================
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n===============================")
print("Model Performance")
print("===============================")
print("Mean Absolute Error :", round(mae,2))
print("Mean Squared Error  :", round(mse,2))
print("Root Mean Squared Error :", round(rmse,2))
print("R2 Score :", round(r2,4))

# ==========================
# Plot Training & Validation Loss
# ==========================
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.grid(True)

plt.show()

# ==========================
# Actual vs Predicted Plot
# ==========================
plt.figure(figsize=(6,6))

plt.scatter(y_test,
            y_pred,
            color='blue')

plt.xlabel("Actual House Price")

plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Price")

plt.grid(True)

plt.show()

# ==========================
# Save Model
# ==========================
model.save("HousePrice_ANN_Experiment2.keras")

print("\nModel Saved Successfully.")