# 🔍 Text Classification & Semantic Article Recommendation

This project is a Streamlit-based web application that combines **text classification** using a fine-tuned BERT model and **semantic article recommendations** powered by Sentence-BERT. It is designed to assist users in quickly identifying the topic of a given text and retrieving relevant articles from a labeled dataset.


### 🧾 Page 1: Text Classification + Similar Article Recommendation

- 🔤 **Input** any free-form text.
- 🤖 **Classify** the input using a fine-tuned `bert-base-uncased` model trained on your labeled dataset.
- 🎯 **Filter** the dataset based on the predicted label.
- 🧠 **Recommend Top-3 Semantically Similar Articles** using Sentence-BERT (`all-MiniLM-L6-v2`) and cosine similarity.
- 📌 **Display** article title, URL, semantic similarity score, and a short summary of each recommended article.

### 📊 Page 2: Model Evaluation Dashboard

- 📈 Visualize the classification model’s **performance metrics** such as:
  - Accuracy
  - F1 Score
- 🔍 Display and inspect:
  - Model configuration
  - Number of training epochs
  - Number of predicted classes
- 📤 Easily understand model strengths and limitations.

## 📂 Project Structure

- `labeled data.csv`: Your original labeled dataset with columns like `content`, `predict`, `title`, `url`.
- `bert_model/`: Folder containing your fine-tuned classification model(too big can not upload to github).
- `app.py`: Main Streamlit app file.
- `embedding_generator.py`: Script to generate and save content embeddings.
- `labeled_data_with_embeddings.csv`: Final dataset with precomputed Sentence-BERT vectors.


