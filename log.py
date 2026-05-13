import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_iris()

X = data.data[:, :1]
y = data.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Accuracy
print("Accuracy:", model.score(X, y))

# Probability curve
x = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_prob = model.predict_proba(x)[:,0]

# Plot
plt.scatter(X, y)
plt.plot(x, y_prob)

plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression")

plt.show()