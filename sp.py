
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train model
model = Perceptron()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Results
print("Accuracy:", accuracy_score(y, y_pred))
print("Weights:", model.coef_)
print("Bias:", model.intercept_)



#Multilayer Perceptron
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()

X = data.data
y = data.target

# Multilayer Perceptron model
model = MLPClassifier(
    hidden_layer_sizes=(5,),
    max_iter=3000,
    random_state=42
)

# Train model
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Results
print("Accuracy:", accuracy_score(y, y_pred))
print("Weights:", model.coefs_)
print("Bias:", model.intercepts_)