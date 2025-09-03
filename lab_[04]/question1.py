import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\SML\lab_[04]\bluegills.txt", sep="\t")   
print(data.head())         
print(data.columns)
X = data['age'].values
Y = data['length'].values


x_mean = np.mean(X)
y_mean = np.mean(Y)


numerator = np.sum((X - x_mean) * (Y - y_mean))
denominator = np.sum((X - x_mean) ** 2)

beta1 = numerator / denominator
beta0 = y_mean - beta1 * x_mean

print("Slope (β1):", beta1)
print("Intercept (β0):", beta0)
print(f"Regression Equation: y = {beta0:.2f} + {beta1:.2f}x")

age_input = 10
predicted_length = beta0 + beta1 * age_input
print(f"Predicted length at age {age_input}: {predicted_length:.2f}")

plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, beta0 + beta1 * X, color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Length')
plt.title('Age vs Length Regression')
plt.legend()
plt.show()
