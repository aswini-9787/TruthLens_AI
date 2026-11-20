
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Load datasets

print("Loading datasets...")

true_data = pd.read_csv("dataset/True.csv", encoding="latin-1")
fake_data = pd.read_csv("dataset/Fake.csv", encoding="latin-1")

# STEP 2: Assign labels

true_data["label"] = "REAL"
fake_data["label"] = "FAKE"

# STEP 3: Combine datasets

data = pd.concat([true_data, fake_data], ignore_index=True)

# STEP 4: Prepare news text

data = data.fillna("")

data["news"] = data["title"] + " " + data["text"]

data = data[["news", "label"]]
data = data.drop_duplicates(subset=["news"])

print("Total articles:", len(data))

# STEP 5: Separate input and output

X = data["news"]
y = data["label"]

# STEP 6: Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))

# STEP 7: Create Machine Learning pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# STEP 8: Train the model

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

# STEP 9: Test the model

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# STEP 10: Save the trained model

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")

print("\nModel saved successfully!")
print("Location: model/fake_news_model.pkl")
