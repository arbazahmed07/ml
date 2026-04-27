
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2,4,5,4,6])

model = LinearRegression()
model.fit(X, y)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Linear Regression")
plt.show()