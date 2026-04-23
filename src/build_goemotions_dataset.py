import pandas as pd
from datasets import load_dataset
import os

# GoEmotions 27 emotion categories (indices 0-26)
EMOTION_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Indices of regret‑related emotions (positive examples)
REGRET_INDICES = {
    EMOTION_NAMES.index("remorse"),        # 24
    EMOTION_NAMES.index("embarrassment"),  # 12
    EMOTION_NAMES.index("disappointment"), # 9
    EMOTION_NAMES.index("sadness"),        # 25
    EMOTION_NAMES.index("grief")           # 16
}

# Indices of safe emotions (negative examples)
SAFE_INDICES = {
    EMOTION_NAMES.index("neutral"),        # 27
    EMOTION_NAMES.index("admiration"),     # 0
    EMOTION_NAMES.index("amusement"),      # 1
    EMOTION_NAMES.index("approval"),       # 4
    EMOTION_NAMES.index("joy"),            # 17
    EMOTION_NAMES.index("love"),           # 18
    EMOTION_NAMES.index("excitement")      # 13
}

def map_labels_to_regret(labels):
    """
    Given a list of integer labels, returns:
    1 -> if any regret label is present (positive)
    0 -> if only safe labels (and no regret) are present
    None -> otherwise (mixed undecided or other emotions)
    """
    has_regret = any(l in REGRET_INDICES for l in labels)
    has_safe = any(l in SAFE_INDICES for l in labels)
    
    if has_regret:
        return 1
    elif has_safe:
        return 0
    else:
        return None

def process_goemotions_dataset():
    print("Loading GoEmotions dataset (simplified)...")
    dataset = load_dataset("go_emotions", "simplified")
    
    all_data = []
    for split in ['train', 'validation', 'test']:
        print(f"Processing {split} split...")
        for example in dataset[split]:
            labels = example['labels']   # list of integers
            label_value = map_labels_to_regret(labels)
            if label_value is not None:
                all_data.append({
                    'text': example['text'],
                    'label': label_value
                })
    
    if not all_data:
        print("Warning: No data samples matched. Please check label indices.")
        return
    
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs("./data", exist_ok=True)
    df.to_csv("./data/reddit_regret_data.csv", index=False)
    
    print(f"Dataset saved to './data/reddit_regret_data.csv'")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples (regret): {(df['label'] == 1).sum()}")
    print(f"Negative samples (non-regret): {(df['label'] == 0).sum()}")

if __name__ == "__main__":
    process_goemotions_dataset()