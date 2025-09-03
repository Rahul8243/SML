import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("headbrain.csv")  
XY = data[["Head Size(cm^3)", "Brain Weight(grams)"]]

# Split into X and Y
X = XY[["Head Size(cm^3)"]].values  # keep 2D shape for sklearn
Y = XY["Brain Weight(grams)"].values

# Train-test split (same as Section 1 → 75% train, 25% test)
split_index = int(0.75 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# 2. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Get slope (coef_) and intercept
print("Slope (b1):", model.coef_[0])
print("Intercept (b0):", model.intercept_)

# 3. Predict on test set
Y_pred_test = model.predict(X_test)

# 4. Calculate RMSE and R²
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
r2 = r2_score(Y_test, Y_pred_test)

print("\nRMSE (inbuilt):", rmse)
print("R-squared (inbuilt):", r2)

# Plot regression line with test data
plt.scatter(X_test, Y_test, color="blue", label="Test Data")
plt.plot(X_test, Y_pred_test, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Head Size (cm^3)")
plt.ylabel("Brain Weight (grams)")
plt.title("Head Size vs Brain Weight (Test Data)")
plt.legend()
plt.show()
plt.savefig("head_size_vs_brain_weight.png")
