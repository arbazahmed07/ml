import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Reviews.csv")

# Use review text and score
data = data[["Text", "Score"]].dropna()

# Convert score to sentiment
data["Sentiment"] = data["Score"].apply(
    lambda x: "Positive" if x > 3 else "Negative"
)

# Features and target
X = data["Text"]
y = data["Sentiment"]

# Convert text to numbers
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))