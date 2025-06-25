import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

#  Page setting
st.set_page_config(page_title="Text Classification & Recommendations", layout="wide")
st.title("🔍 Auto Content Tagging & Similar Articles Recommendations")

#  Load the classification model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert_model")
tokenizer = AutoTokenizer.from_pretrained("bert_model")

#  Tag mapping
with open("bert_model/label2id.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

df = pd.read_csv("labeled data.csv")

# Sentence-BERT
embedder = SentenceTransformer("all-MiniLM-L6-v2")

user_text = st.text_area("✏️ Input your content:", height=200)

if st.button("🔮 Predict & Recommend"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("🤖 Predicting label..."):
            inputs = tokenizer(user_text, return_tensors="pt")
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predicted_label = id2label[pred]

        st.success(f"🎯 Predicted Label: **{predicted_label}**")

        #  Filter data from the same label.
        subset = df[df["predict"] == predicted_label].copy()

        if subset.empty:
            st.info("⚠️ No similar content found under this label.")
        else:
            with st.spinner("🔍 Calculating semantic similarity..."):
                # ✅ Sentence-BERT
                emb_user = embedder.encode(user_text, convert_to_tensor=True)
                emb_corpus = embedder.encode(subset["content"].tolist(), convert_to_tensor=True)

                #  Similarity calculation
                cos_scores = util.pytorch_cos_sim(emb_user, emb_corpus)[0].cpu().numpy()
                top_indices = cos_scores.argsort()[-3:][::-1]
                recommended = subset.iloc[top_indices].copy()

            st.markdown("### 📚 Recommended Articles:")
            for i, (idx, row) in enumerate(recommended.iterrows()):
                sim_score = cos_scores[top_indices[i]]
                st.markdown(f"-  Titles: **{row['title']}**  ")

                st.markdown(f"  🧠 Similarity Score: `{sim_score:.3f}`")
                if 'content' in row and isinstance(row['content'], str):
                    summary = row['content'][:150].strip().replace('\n', ' ') + "..."
                    st.markdown(f"  📝 Abstract: {summary}")
                st.markdown(f"     🔗 [Link]({row['url']})  ")


