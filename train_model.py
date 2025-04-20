# train_model.py
import pandas as pd
import string
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv("Amazon-Product-Reviews - Amazon Product Review (1).csv")

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["review_body"] = df["review_body"].astype(str).apply(clean_text)
df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
df = df.dropna(subset=["sentiment", "review_body"])

X_train, X_test, y_train, y_test = train_test_split(df["review_body"], df["sentiment"], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "sentiment_model.pkl")

os.makedirs("static", exist_ok=True)
sentiment_counts = df['sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['tomato', 'seagreen'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment (0 = Negative, 1 = Positive)")
plt.ylabel("Number of Reviews")
plt.savefig("chart.png")
plt.close()

print("✅ Model trained and chart saved.")
