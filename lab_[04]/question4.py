import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\SML\SML\lab_[04]\bluegills.txt", sep="\t")
X = data[['age']].values
y = data['length'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict on test data
y_pred = linear_model.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)

print(f"R² Score between Age and Length: {r2:.4f}")
