import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df)
cluster_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
df['variety'] = df['cluster'].map(cluster_mapping)
new_sample = pd.DataFrame([[5.8, 2.7, 5.1, 1.9]], columns=iris.feature_names)
predicted_cluster = kmeans.predict(new_sample)[0]
predicted_variety = cluster_mapping[predicted_cluster]
print(f"Predicted Cluster: {predicted_cluster}")
print(f"Prediction for new sample: {predicted_variety}")
plt.figure(figsize=(8, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['cluster'], cmap='viridis', edgecolor='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-Means Clustering on Iris Dataset")
plt.show()
