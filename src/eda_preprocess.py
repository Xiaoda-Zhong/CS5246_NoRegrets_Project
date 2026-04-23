import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

df = pd.read_csv("./data/analyzed_data.csv")

def parse_toxic_categories(cat_str):
    try:
        return json.loads(cat_str.replace("'", '"'))
    except:
        return {}

df['toxic_dict'] = df['toxic_categories'].apply(parse_toxic_categories)
for cat in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    df[f'prob_{cat}'] = df['toxic_dict'].apply(lambda d: d.get(cat, 0.0) if isinstance(d, dict) else 0.0)

plt.figure(figsize=(8,5))
sns.boxplot(x='original_label', y='toxicity_score', data=df)
plt.title('Toxicity Score Distribution by Regret Label')
plt.xticks([0,1], ['Non-Regret', 'Regret'])
os.makedirs("./data", exist_ok=True)
plt.savefig("./data/toxicity_boxplot.png")
plt.close()

cat_cols = ['prob_toxic', 'prob_severe_toxic', 'prob_obscene', 'prob_threat', 'prob_insult', 'prob_identity_hate']
mean_by_label = df.groupby('original_label')[cat_cols].mean()
mean_by_label.T.plot(kind='bar', figsize=(10,6))
plt.title('Average Toxicity Probability by Subcategory')
plt.ylabel('Mean Probability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./data/toxicity_subcategory_bars.png")
plt.close()

print("EDA plots saved to ./data/")
print(f"Dataset size: {len(df)}")
print(f"Regret samples: {(df['original_label']==1).sum()}")
print(f"Non-regret samples: {(df['original_label']==0).sum()}")