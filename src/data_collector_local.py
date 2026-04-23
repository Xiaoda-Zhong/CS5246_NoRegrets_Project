import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load model and tokenizer once
MODEL_NAME = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Labels for toxic-bert: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def analyze_text_local(text):
    """Return sentiment (hardcoded as unknown) and toxicity scores using toxic-bert."""
    # Truncate to 512 tokens (BERT limit)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy()[0]  # multi-label probabilities
    
    toxicity_dict = {label: float(prob) for label, prob in zip(LABELS, probs)}
    overall_toxicity = float(np.mean(probs))  # average over all categories
    
    # sentiment is not provided by toxic-bert, we set to 'unknown'
    sentiment = 'unknown'
    return sentiment, overall_toxicity, toxicity_dict

def process_dataset_local(input_csv_path, output_csv_path):
    if not os.path.exists(input_csv_path):
        print(f"Error: {input_csv_path} not found.")
        return

    df_input = pd.read_csv(input_csv_path)
    results = []
    total = len(df_input)

    for idx, row in df_input.iterrows():
        raw_text = row['text']
        sentiment, toxicity_score, toxicity_categories = analyze_text_local(raw_text)
        
        results.append({
            'original_label': row['label'],
            'sentiment': sentiment,
            'toxicity_score': toxicity_score,
            'toxic_categories': json.dumps(toxicity_categories),
            'text': raw_text
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{total}")

    df_output = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_output.to_csv(output_csv_path, index=False)
    print(f"Analysis complete. Saved to {output_csv_path}")

if __name__ == "__main__":
    csv_path = "./data/reddit_regret_data.csv"
    output_path = "./data/analyzed_data.csv"
    process_dataset_local(csv_path, output_path)