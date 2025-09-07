import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\SML\SML\lab_[04]\bluegills.txt", sep="\t")   # tab separated
X = data[['age']].values
y = data['length'].values

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_errors = []
test_errors = []
degrees = range(1, 11)

# Loop for polynomial degrees 1 → 10
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # RMS error
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

# Plot Train vs Test Error
plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, test_errors, marker='s', label="Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("RMS Error")
plt.title("Polynomial Regression: Training vs Testing Error")
plt.legend()
plt.grid(True)
plt.show()
