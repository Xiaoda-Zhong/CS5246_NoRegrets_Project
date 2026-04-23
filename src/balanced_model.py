import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
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

# 1. Logistic Regression with class_weight='balanced'
print("=== Logistic Regression with balanced weights ===")
lr_bal = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_bal.fit(X_train, y_train)
y_pred_lr = lr_bal.predict(X_test)
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

# 2. Random Forest with class_weight='balanced'
print("\n=== Random Forest with balanced weights ===")
rf_bal = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_bal.fit(X_train, y_train)
y_pred_rf = rf_bal.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# 3. SMOTE oversampling + Random Forest
print("\n=== SMOTE + Random Forest ===")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)
print(classification_report(y_test, y_pred_smote))
print(confusion_matrix(y_test, y_pred_smote))


importances = rf_smote.feature_importances_
print("\nFeature importances (SMOTE + RF):")
for name, imp in zip(feature_cols, importances):
    print(f"{name}: {imp:.4f}")