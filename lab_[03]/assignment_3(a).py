import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\SML\\lab_[03]\\headbrain.csv")  
XY = data[["Head Size(cm^3)", "Brain Weight(grams)"]]

# Preview data
print("First 5 rows of XY:\n", XY.head())

# Statistical summary
print("\nStatistical summary:\n", XY.describe())

# 3. Split XY into X and Y
X = XY["Head Size(cm^3)"].values    
Y = XY["Brain Weight(grams)"].values 

# Print the shape of X and Y
print("\nShape of X:", X.shape)
print("Shape of Y:", Y.shape)

# 4. Check for missing values
print("\nMissing values in X:", np.isnan(X).sum())
print("Missing values in Y:", np.isnan(Y).sum())

# 5. Calculate the mean of X and Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)
print("\nMean of X (Head Size):", mean_X)
print("Mean of Y (Brain Weight):", mean_Y)

# 6. Fill missing values with the calculated mean (if any)
X = np.where(np.isnan(X), mean_X, X)
Y = np.where(np.isnan(Y), mean_Y, Y)

print("\nAfter filling missing values:")
print("Missing values in X:", np.isnan(X).sum())
print("Missing values in Y:", np.isnan(Y).sum())

# 7. Train-Test Split (75% train, 25% test)
split_index = int(0.75 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

print("\nTraining set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# 8. Implement Linear Regression from scratch
# Formula: Y = b0 + b1*X

# Calculate slope (b1) and intercept (b0)
num = np.sum((X_train - np.mean(X_train)) * (Y_train - np.mean(Y_train)))
den = np.sum((X_train - np.mean(X_train)) ** 2)
b1 = num / den
b0 = np.mean(Y_train) - b1 * np.mean(X_train)

print("\nSlope (b1):", b1)
print("Intercept (b0):", b0)

# Predict Y values for training set
Y_pred_train = b0 + b1 * X_train

# Plot scatter and regression line
plt.scatter(X_train, Y_train, color="blue", label="Train Data")
plt.plot(X_train, Y_pred_train, color="red", label="Regression Line")
plt.xlabel("Head Size (cm^3)")
plt.ylabel("Brain Weight (grams)")
plt.title("Head Size vs Brain Weight (Train Data)")
plt.legend()
plt.show()
plt.savefig("regression_plot.png")  # Save the plot as an image file

# 9. RMSE on test set
Y_pred_test = b0 + b1 * X_test
rmse = np.sqrt(np.mean((Y_test - Y_pred_test) ** 2))
print("\nRoot Mean Square Error (RMSE):", rmse)

# 10. R-squared value
ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
ss_res = np.sum((Y_test - Y_pred_test) ** 2)
r2 = 1 - (ss_res / ss_total)
print("R-squared (R²):", r2)

split_index = int(0.75 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

