import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Shape of the dataset:", df.shape)

# Rename columns
df = df.rename(columns={
    'sepal length (cm)': 'SepalLength',
    'sepal width (cm)': 'SepalWidth',
    'petal length (cm)': 'PetalLength',
    'petal width (cm)': 'PetalWidth'
})

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Boxplot example
plt.figure(figsize=(6,4))
sns.boxplot(y=df['PetalLength'])
plt.title('Distribution of PetalLength')
plt.ylabel('PetalLength')
plt.show()

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Scaling features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFirst 5 rows of scaled training data:")
print(pd.DataFrame(X_train_scaled, columns=features).head())

# Step 7: Train Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = nb_model.predict(X_test_scaled)


# Step 8: Confusion Matrix + Heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Naive Bayes)")
plt.show()

# Step 9: Accuracy & Recall
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')

print("\nAccuracy:", round(accuracy, 2))
print("Recall (macro-average):", round(recall, 2))

# Step 10: Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
