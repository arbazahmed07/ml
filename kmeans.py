import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv("Iris.csv")
X = data.iloc[:, :2]   # take only 2 features for easy plotting

# Model
model = KMeans(n_clusters=3)
model.fit(X)

# Plot
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=model.labels_)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1])
plt.title("K-Means Clustering")
plt.show()