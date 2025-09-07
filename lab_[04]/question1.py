import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------
# Step 1: Load Dataset
# -------------------
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\SML\SML\lab_[04]\bluegills.txt", sep="\t")
print("First few rows of dataset:\n", data.head())

# Features (X) and Target (y)
X = data[['age']].values   # Independent variable (age)
y = data['length'].values  # Dependent variable (length)

# -------------------
# Step 2: Train Linear Regression Model
# -------------------
linear_model = LinearRegression()
linear_model.fit(X, y)

# Predictions
y_pred = linear_model.predict(X)

# -------------------
# Step 3: Evaluate Model
# -------------------
print("\nLinear Regression Results:")
print("Coefficient (slope):", linear_model.coef_[0])
print("Intercept:", linear_model.intercept_)
print("R² Score:", r2_score(y, y_pred))
print("Mean Squared Error:", mean_squared_error(y, y_pred))

# -------------------
# Step 4: Plot
# -------------------
# Sort X for smooth line
X_sorted = np.sort(X, axis=0)
y_pred_sorted = linear_model.predict(X_sorted)

plt.scatter(X, y, color='blue', label="Data Points")
plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label="Linear Regression Line")
plt.xlabel("Age")
plt.ylabel("Length")
plt.title("Bluegills - Linear Regression")
plt.legend()
plt.show()
