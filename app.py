import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# Streamlit UI
st.set_page_config(page_title="Text Classification Present", layout="wide")

# load the model map
model = AutoModelForSequenceClassification.from_pretrained("bert_model")
tokenizer = AutoTokenizer.from_pretrained("bert_model")

with open("bert_model/label2id.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
st.title("Based on trained model Tags classification")



text = st.text_area("Input Your Text：", height=200)
if st.button("🔍 Predict:"):

    if not text.strip():
        st.warning("Input Your Text")
    else:
        with st.spinner("Model Inferencing..."):
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            label = id2label[pred]
        st.success(f"Possible Class：**{label}**")
