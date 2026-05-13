import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data = load_diabetes()

X = data.data[:, 2].reshape(-1, 1)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", np.sqrt(mse))

# Plot graph
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Linear Regression")
plt.show()