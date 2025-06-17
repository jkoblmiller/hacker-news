import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import plotly.express as px

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("hn_keybert_cleaned.csv")
    df["tag_text"] = df["tag_text"].fillna("").astype(str)
    return df

df = load_data()
st.title("Interactive Clustering of Hacker News Posts")

# TF-IDF vectorization + distance matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["tag_text"])
dist_matrix = cosine_distances(tfidf_matrix)

# Sidebar: clustering method
method = st.sidebar.selectbox("Choose Clustering Method", ["HDBSCAN", "DBSCAN"])

if method == "HDBSCAN":
    min_cluster_size = st.sidebar.slider("min_cluster_size", 5, 100, 20)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
    cluster_labels = clusterer.fit_predict(dist_matrix)
    probabilities = clusterer.probabilities_
else:
    eps = st.sidebar.slider("eps", 0.1, 1.0, 0.6)
    min_samples = st.sidebar.slider("min_samples", 3, 20, 5)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = clusterer.fit_predict(dist_matrix)
    probabilities = np.ones(len(cluster_labels))

df["cluster"] = cluster_labels
df["confidence"] = probabilities

# t-SNE projection
@st.cache_data
def run_tsne(_matrix):  # underscore avoids caching error
    tsne = TSNE(n_components=2, random_state=42, init="random")
    return tsne.fit_transform(_matrix)

df[["x", "y"]] = run_tsne(tfidf_matrix)

# --- SIDEBAR FILTERS (move this down here!) ---
show_noise = st.sidebar.checkbox("Show noise (cluster = -1)", value=False)
cluster_options = sorted(df["cluster"].unique())
selected_cluster = st.sidebar.selectbox("Filter to specific cluster:", ["All"] + cluster_options)

# Apply cluster filter
if selected_cluster != "All":
    plot_df = df[df["cluster"] == selected_cluster]
elif not show_noise:
    plot_df = df[df["cluster"] != -1]
else:
    plot_df = df.copy()

#These are KeyBERT-generated tags 
# tags into vectors using TF-IDF
# dist_matrix = cosine_distances(tfidf_matrix)
# Plotting
hover_cols = [col for col in ["content", "keybert_tags"] if col in df.columns]
fig = px.scatter(
    plot_df,
    x="x", y="y",
    color="cluster",
    opacity=plot_df["confidence"],
    hover_data=hover_cols,
    title=f"{method} Clusters (Confidence as Opacity)"
)

st.plotly_chart(fig, use_container_width=True)

from keybert import KeyBERT

st.markdown("---")
with st.expander("Upload & Tag Your Own Data (KeyBERT)", expanded=False):
    uploaded_file = st.file_uploader("Upload a CSV file with a 'content' column", type="csv")

    @st.cache_resource
    def load_keybert_model():
        return KeyBERT(model="paraphrase-MiniLM-L3-v2")  # faster
        # KeyBERT(model="all-MiniLM-L6-v2")

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)

        if "content" not in user_df.columns:
            st.error("Your CSV must have a column named 'content'.")
        else:
            kb_model = load_keybert_model()
            st.success("KeyBERT model loaded!")

            with st.spinner("Extracting keywords for each row..."):
                user_df["keybert_tags"] = user_df["content"].astype(str).apply(
                    lambda x: [kw for kw, _ in kb_model.extract_keywords(x, top_n=4)]
                )

            st.subheader("Preview of Tags:")
            st.dataframe(user_df[["content", "keybert_tags"]].head(10))

            csv_out = user_df.to_csv(index=False)
            st.download_button("⬇Download Tagged CSV", csv_out, file_name="keybert_output.csv")
