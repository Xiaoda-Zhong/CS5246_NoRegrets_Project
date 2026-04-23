import pandas as pd
import re
import os

def detect_sensitive_info(text):
    phone = re.search(r'(\+65)?[89]\d{7}', text) is not None
    email = re.search(r'\S+@\S+', text) is not None
    nric = re.search(r'[STFG]\d{7}[A-Z]', text) is not None
    return phone, email, nric

def add_sensitive_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    phones = []
    emails = []
    nrics = []
    for text in df['text']:
        p, e, n = detect_sensitive_info(str(text))
        phones.append(int(p))
        emails.append(int(e))
        nrics.append(int(n))
    df['has_phone'] = phones
    df['has_email'] = emails
    df['has_nric'] = nrics
    df['has_sensitive'] = (df['has_phone'] | df['has_email'] | df['has_nric']).astype(int)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Sensitive features added. Saved to {output_csv}")
    print(f"Proportion with sensitive info: {df['has_sensitive'].mean():.2%}")

if __name__ == "__main__":
    add_sensitive_features("./data/analyzed_data.csv", "./data/analyzed_with_sensitive.csv")