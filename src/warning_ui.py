import gradio as gr
import joblib
import re
import numpy as np

model = joblib.load("./models/tfidf_lr.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")

def desensitize_text(text):
    text = re.sub(r'(\+65)?[89]\d{7}', '[PHONE]', text)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'[STFG]\d{7}[A-Z]', '[NRIC]', text)
    return text

def has_sensitive_info(text):
    phone = re.search(r'(\+65)?[89]\d{7}', text) is not None
    email = re.search(r'\S+@\S+', text) is not None
    nric = re.search(r'[STFG]\d{7}[A-Z]', text) is not None
    return phone or email or nric

def predict_regret(text):
    if not text.strip():
        return "⚠️ Please enter some text.", ""
    
    prob = model.predict_proba(vectorizer.transform([text]))[0]
    regret_prob = prob[1]
    pred = model.predict(vectorizer.transform([text]))[0]
    
    sensitive = has_sensitive_info(text)
    warning = ""
    if pred == 1 or regret_prob > 0.5 or sensitive:
        warning = "⚠️ This post may be regrettable. "
        if sensitive:
            warning += "Contains sensitive information. "
        if regret_prob > 0.5:
            warning += f"Model confidence: {regret_prob:.2f}. "
        warning += "Consider revising."
    else:
        warning = "✅ This post appears safe to publish."
    
    suggested = desensitize_text(text)
    return warning, suggested

with gr.Blocks(title="No Regrets! Warning System") as demo:
    gr.Markdown("# No Regrets! - Pre‑Post Warning System")
    gr.Markdown("Paste your draft below. We'll detect regrettable content and suggest redactions.")
    with gr.Row():
        input_text = gr.Textbox(label="Your Draft", lines=5, placeholder="Type your post here...")
        output_warning = gr.Textbox(label="Warning", lines=2)
        output_suggestion = gr.Textbox(label="Suggested Redaction", lines=5)
    submit_btn = gr.Button("Analyze")
    submit_btn.click(fn=predict_regret, inputs=input_text, outputs=[output_warning, output_suggestion])

demo.launch()