import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load datasets
bank = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\SML\SML\lab_[04]\NIFTY BANK.csv")
energy = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\SML\SML\lab_[04]\NIFTY Energy.csv")

# Extract 'Open' prices
X = bank[['Open']].values   # Independent variable
y = energy['Open'].values   # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 1. Linear Regression
# -----------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression R² Score: {r2_linear:.4f}")

# Visualization (Linear)
plt.scatter(X, y, color='blue', alpha=0.5, label="Data Points")
plt.plot(X, linear_model.predict(X), color='red', label="Linear Fit")
plt.title("Bank Nifty vs Energy Nifty (Linear Regression)")
plt.xlabel("Bank Nifty Open")
plt.ylabel("Energy Nifty Open")
plt.legend()
plt.show()

# -----------------------------
# 2. Polynomial Regression (degree 2 to 5)
# -----------------------------
for degree in range(2, 6):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_train)
    
    y_pred_poly = poly_model.predict(poly.transform(X_test))
    r2_poly = r2_score(y_test, y_pred_poly)
    
    print(f"Polynomial Degree {degree} R² Score: {r2_poly:.4f}")
    
    # Visualization
    X_sorted = np.sort(X, axis=0)
    plt.scatter(X, y, color='blue', alpha=0.5, label="Data Points")
    plt.plot(X_sorted, poly_model.predict(poly.transform(X_sorted)), color='green', label=f"Poly deg {degree}")
    plt.title(f"Bank Nifty vs Energy Nifty (Polynomial deg {degree})")
    plt.xlabel("Bank Nifty Open")
    plt.ylabel("Energy Nifty Open")
    plt.legend()
    plt.show()
