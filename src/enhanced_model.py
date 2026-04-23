import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from textblob import TextBlob
import nltk
import joblib
import os

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

df = pd.read_csv("./data/analyzed_data.csv")

# 解析毒性特征
def parse_categories(cat_str):
    try:
        return json.loads(cat_str.replace("'", '"'))
    except:
        return {}

df['toxic_dict'] = df['toxic_categories'].apply(parse_categories)
for cat in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    df[f'prob_{cat}'] = df['toxic_dict'].apply(lambda d: d.get(cat, 0.0) if isinstance(d, dict) else 0.0)

# 额外特征提取
def has_sensitive(text):
    phone = bool(re.search(r'(\+65)?[89]\d{7}', text))
    email = bool(re.search(r'\S+@\S+', text))
    nric = bool(re.search(r'[STFG]\d{7}[A-Z]', text))
    return int(phone or email or nric)

def text_length(text):
    return len(text)

def polarity(text):
    return TextBlob(text).sentiment.polarity

df['has_sensitive'] = df['text'].apply(has_sensitive)
df['text_len'] = df['text'].apply(text_length)
df['polarity'] = df['text'].apply(polarity)

feature_cols = ['toxicity_score', 'prob_toxic', 'prob_severe_toxic', 'prob_obscene',
                'prob_threat', 'prob_insult', 'prob_identity_hate',
                'has_sensitive', 'text_len', 'polarity']
X = df[feature_cols].fillna(0)
y = df['original_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

importances = rf.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{name}: {imp:.4f}")


os.makedirs("./models", exist_ok=True)
joblib.dump(rf, "./models/enhanced_rf.pkl")
with open("./models/feature_cols.txt", "w") as f:
    f.write("\n".join(feature_cols))
print("Model saved to ./models/")