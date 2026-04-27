import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv("spine.csv")

# Encode categorical columns to numeric
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col])

# Split into features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# -------- Classification (Logistic Regression) --------
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test, y_pred))

# -------- Regression (Linear Regression) --------
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)
print("Regression MSE:", mean_squared_error(y_test, y_pred_reg))

# -------- Clustering (KMeans) --------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Evaluate clustering quality using Silhouette Score
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Compute cluster purity using true labels
contingency_matrix = pd.crosstab(y, labels)
purity = np.sum(np.max(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

print("Purity:", purity)