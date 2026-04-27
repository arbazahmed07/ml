import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load data
iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier()

# -------- HOLDOUT --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train)
print("Holdout Accuracy:", model.score(X_test, y_test))

# -------- K-FOLD --------
scores = cross_val_score(model, X, y, cv=5)
print("K-Fold Avg Accuracy:", scores.mean())

# -------- BOOTSTRAP --------
n = len(X)
idx = np.random.choice(n, n, replace=True)

X_train, y_train = X[idx], y[idx]
mask = ~np.isin(range(n), idx)
X_test, y_test = X[mask], y[mask]

model.fit(X_train, y_train)
print("Bootstrap Accuracy:", model.score(X_test, y_test))