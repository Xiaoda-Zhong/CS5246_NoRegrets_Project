import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

df = pd.read_csv("./data/analyzed_data.csv")

def parse_categories(cat_str):
    try:
        return json.loads(cat_str.replace("'", '"'))
    except:
        return {}

df['toxic_dict'] = df['toxic_categories'].apply(parse_categories)
for cat in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    df[f'prob_{cat}'] = df['toxic_dict'].apply(lambda d: d.get(cat, 0.0) if isinstance(d, dict) else 0.0)

feature_cols = ['toxicity_score', 'prob_toxic', 'prob_severe_toxic', 'prob_obscene', 
                'prob_threat', 'prob_insult', 'prob_identity_hate']
X = df[feature_cols].fillna(0)
y = df['original_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("=== Logistic Regression ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n=== Random Forest ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

importances = rf.feature_importances_
print("\nFeature importances (Random Forest):")
for name, imp in zip(feature_cols, importances):
    print(f"{name}: {imp:.4f}")