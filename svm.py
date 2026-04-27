import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Data
X, y = make_blobs(n_samples=100, centers=2, random_state=1)

# Model
model = SVC(kernel='linear')
model.fit(X, y)

# Plot
plt.scatter(X[:,0], X[:,1], c=y)

# Decision boundary
w = model.coef_[0]
b = model.intercept_[0]
x = np.linspace(X[:,0].min(), X[:,0].max())
y_line = -(w[0]*x + b)/w[1]

plt.plot(x, y_line)
plt.title("Linear SVM")
plt.show()




#Non linear SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

# Data
X, y = make_circles(n_samples=100, noise=0.1)

# Model
model = SVC(kernel='rbf')
model.fit(X, y)

# Plot
plt.scatter(X[:,0], X[:,1], c=y)

# Grid for boundary
xx, yy = np.meshgrid(np.linspace(-1.5,1.5,100),
                     np.linspace(-1.5,1.5,100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title("Non-Linear SVM")
plt.show()