# 🔍 Text Classification & Semantic Article Recommendation

This project is a Streamlit-based web application that combines **text classification** using a fine-tuned BERT model and **semantic article recommendations** powered by Sentence-BERT. It is designed to assist users in quickly identifying the topic of a given text and retrieving relevant articles from a labeled dataset.


### 🧾 Page 1: Text Classification + Similar Article Recommendation

- 🔤 **Input** any free-form text.
- 🤖 **Classify** the input using a fine-tuned `bert-base-uncased` model trained on your labeled dataset.
- 🎯 **Filter** the dataset based on the predicted label.
- 🧠 **Recommend Top-3 Semantically Similar Articles** using Sentence-BERT (`all-MiniLM-L6-v2`) and cosine similarity.
- 📌 **Display** article title, URL, semantic similarity score, and a short summary of each recommended article.

### 📊 Page 2: Semantic Clustering & Interactive Visualization

This page offers an interactive way to explore how news posts (or texts) are distributed in semantic space using precomputed Sentence-BERT embeddings.

#### Key Features:
- 🧠 **Sentence Embeddings**: Based on `all-MiniLM-L6-v2` via Sentence-BERT.
- 🧬 **UMAP Projection**: Reduces high-dimensional embeddings to 2D or 3D for visualization.
- 🧩 **Clustering Options**: Choose between DBSCAN and HDBSCAN to discover latent semantic clusters.
- 🎨 **Coloring Modes**:
  - By predicted topic label
  - By clustering result
- 🕵️‍♀️ **Interactivity**:
  - Hover on points to view title, confidence, and score
  - Type keywords to search article titles
  - Explore articles in a table with metadata
- 🔷 **Convex Hulls**: In 2D, clusters are outlined with dashed boundaries for clarity.

## 📂 Project Structure

- `labeled data.csv`: Your original labeled dataset with columns like `content`, `predict`, `title`, `url`.
- `bert_model/`: Folder containing your fine-tuned classification model(too big can not upload to github).
- `pages.py`: Main Streamlit app file.
- `embedding_generator.py`: Script to generate and save content embeddings.
- `labeled_data_with_embeddings.csv`: Final dataset with precomputed Sentence-BERT vectors.


