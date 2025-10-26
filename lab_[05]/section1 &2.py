import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Section 1: Data Preparation
df = pd.read_csv('lab_[05]/diabetes.csv')
print("Original Data (head):")
print(df.head())

df = df.rename(columns={
    'Pregnancies': 'Preg',
    'BloodPressure': 'BP',
    'SkinThickness': 'Skin',
    'DiabetesPedigreeFunction': 'Pedigree'
})
print("\nModified Data (head):")
print(df.head())

print("\nData Types:")
print(df.dtypes)

features = ['Preg', 'BP', 'Insulin', 'BMI', 'Pedigree', 'Age']
target = 'Outcome'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(8,5))
sns.scatterplot(x=X_train['BMI'], y=X_train['Pedigree'], hue=y_train)
plt.title('BMI vs Pedigree (Train Data)')
plt.xlabel('BMI')
plt.ylabel('Pedigree')
plt.show()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Section 2: K-Nearest Neighbor Model

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}")
print(f"F1 Score: {f1:.3f}")

# Classification report with custom target labels
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
