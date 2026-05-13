import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load inbuilt Iris dataset
data = load_iris()

# Take first 2 features
X = data.data[:, :2]

# K-Means model
model = KMeans(n_clusters=3, random_state=42)

# Train model
model.fit(X)

# Plot clusters
plt.scatter(X[:,0], X[:,1], c=model.labels_)

# Plot centroids
plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")

plt.show()