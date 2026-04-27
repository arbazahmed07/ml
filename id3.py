import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

# Model (ID3 = entropy)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("ID3 Accuracy:", accuracy_score(y_test, y_pred))

# Plot tree
plt.figure(figsize=(12, 8))
plot_tree(model)
plt.show()