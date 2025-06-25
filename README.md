# 🔍 Text Classification & Semantic Article Recommendation

This project provides a simple and interactive web interface (via Streamlit) to:
- Classify input text using a fine-tuned BERT model.
- Recommend top-3 semantically similar articles based on the predicted label using Sentence-BERT embeddings.

## 🧠 Features

- 🤖 **Text Classification** using a fine-tuned `bert-base-uncased` model.
- 📚 **Semantic Recommendations** using `all-MiniLM-L6-v2` SentenceTransformer.
- 🎯 Filter by predicted class to recommend relevant articles.
- 🔗 Displays article **title**, **URL**, **semantic score**, and a brief **summary**.
- 🌐 Built with **Streamlit** multi-page UI for ease of use.

## 📂 Project Structure

- `labeled data.csv`: Your original labeled dataset with columns like `content`, `predict`, `title`, `url`.
- `bert_model/`: Folder containing your fine-tuned classification model(too big can not upload to github).
- `app.py`: Main Streamlit app file.
- `embedding_generator.py`: Script to generate and save content embeddings.
- `labeled_data_with_embeddings.csv`: Final dataset with precomputed Sentence-BERT vectors.

```bash
python3 -m venv hn_env
source hn_env/bin/activate  # On macOS/Linux

