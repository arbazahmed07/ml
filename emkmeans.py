import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

# Load data
X = load_iris().data

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(X)

# EM (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
g_labels = gmm.fit_predict(X)

# Evaluation using silhouette score
k_score = silhouette_score(X, k_labels)
g_score = silhouette_score(X, g_labels)

# Plot
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=k_labels)
plt.title("K-Means")

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=g_labels)
plt.title("EM (GMM)")

plt.show()

# Comparison
print("K-Means Score:", k_score)
print("EM Score:", g_score)

if g_score > k_score:
    print("EM gives better clustering.")
else:
    print("K-Means gives better clustering.")