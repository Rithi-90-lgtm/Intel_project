import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['variety'] = iris.target
X = df.drop(columns=['variety'])
y = df['variety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy:.2f}")
new_sample = pd.DataFrame([[5.8, 2.7, 5.1, 1.9]], columns=iris.feature_names)
predicted_class = rf.predict(new_sample)[0]
predicted_variety = iris.target_names[predicted_class]
print(f"Prediction for new sample: {predicted_variety}")
plt.figure(figsize=(8, 6))
plt.barh(iris.feature_names, rf.feature_importances_, color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.show()
