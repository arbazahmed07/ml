import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [2,4,5,4,6]
n = len(x)

# Formula
m = (n*sum([x[i]*y[i] for i in range(n)]) - sum(x)*sum(y)) / (n*sum([i*i for i in x]) - sum(x)**2)
c = (sum(y) - m*sum(x)) / n

# Prediction
y_pred = [m*i + c for i in x]

print("m:", m, "c:", c)

# Plot
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.show()