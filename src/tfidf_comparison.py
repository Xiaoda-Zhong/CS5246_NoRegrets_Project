import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("./data/reddit_regret_data.csv")
X = df['text'].fillna('')
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_tfidf, y_train)
y_pred = lr.predict(X_test_tfidf)
print("=== TF‑IDF + Logistic Regression (balanced) ===")
print(classification_report(y_test, y_pred))


joblib.dump(lr, "./models/tfidf_lr.pkl")
joblib.dump(vec, "./models/tfidf_vectorizer.pkl")