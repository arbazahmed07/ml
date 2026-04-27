import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])

# Train model
model = LogisticRegression().fit(X, y)

# Accuracy
print("Accuracy:", model.score(X, y))

# Sigmoid curve
x = np.linspace(1,6,100).reshape(-1,1)
y_prob = model.predict_proba(x)[:,1]

# Plot
plt.scatter(X, y)
plt.plot(x, y_prob)
plt.show()