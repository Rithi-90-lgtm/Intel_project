import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["variety"] = iris.target
df["variety"] = df["variety"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
X = df.drop(columns=["variety"])
y = df["variety"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Model Accuracy: {accuracy:.2f}")
new_sample = pd.DataFrame([[5.8, 2.7, 5.1, 1.9]], columns=iris.feature_names)
predicted_variety = model.predict(new_sample)
print(f"Prediction for new sample: {predicted_variety[0]}")
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.show()
