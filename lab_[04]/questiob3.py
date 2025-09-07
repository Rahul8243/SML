import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\SML\SML\lab_[04]\bluegills.txt", sep="\t")
X = data[['age']].values
y = data['length'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_test_pred_linear = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, y_test_pred_linear)

# Polynomial Regression (degree=3)
degree = 3
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_test_pred_poly = poly_model.predict(X_test_poly)
poly_mse = mean_squared_error(y_test, y_test_pred_poly)

print(f"Linear Regression MSE: {linear_mse:.4f}")
print(f"Polynomial Regression (degree={degree}) MSE: {poly_mse:.4f}")
