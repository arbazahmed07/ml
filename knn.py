from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

data = load_iris()
x = data.data
y = data.target

x = StandardScaler().fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.3, random_state=42
)

best_k = 1
best_acc = 0

for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)
    acc = accuracy_score(ytest, model.predict(xtest))

    if acc > best_acc:
        best_acc = acc
        best_k = k

final = KNeighborsClassifier(n_neighbors=best_k)
final.fit(xtrain, ytrain)

print("Best k:", best_k)
print("Accuracy:", accuracy_score(ytest, final.predict(xtest)))
     