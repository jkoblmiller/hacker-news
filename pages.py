import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap.umap_ as umap
import numpy as np
import os
import ast
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import ConvexHull

st.set_page_config(page_title="Hacker News", layout="wide")
page = st.sidebar.radio("📄 Select Page:", ["Auto Tagging & Recommendations", "Semantic Clustering"])
if page == "Auto Tagging & Recommendations":
    st.title("🔍 Auto Content Tagging & Similar Articles Recommendations")

    #  Load the classification model and tokeni zer
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

elif page == "Semantic Clustering":
    st.title("📊 Interactive Semantic Clustering of Hacker News Posts")
    st.markdown("""
    This interactive dashboard visualizes the semantic space of Hacker News posts based on their content.

    **How it works:**
    - Posts were embedded using a BERT-based model (`all-MiniLM-L6-v2`), producing 384-dimensional vectors.
    - UMAP reduces these embeddings into **2D** or **3D** for visualization.
    - You can apply **density-based clustering** using either DBSCAN or HDBSCAN.
    - Toggle between coloring by **category** (model-predicted topics) or **cluster ID**.
    - **Convex hulls** show the boundary of each detected cluster.
    - Hover or search to inspect individual posts with metadata.
    """)


    # Load data with embedded vectors
    @st.cache_data
    def load_data():
        df = pd.read_csv("labeled_data_with_embeddings.csv")
        df["embedding"] = df["embedding"].apply(ast.literal_eval)
        return df


    df = load_data()

    # Sidebar: category filter
    all_categories = sorted(df["predict"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Filter by Category:", all_categories, default=all_categories)
    filtered_df = df[df["predict"].isin(selected_categories)].copy()

    # Sidebar: color mode toggle
    color_mode = st.sidebar.radio("Color points by:", ["Category", "Cluster"], index=0)

    # Sidebar: 2D vs 3D toggle
    dimensionality = st.sidebar.radio("UMAP Projection:", ["2D", "3D"], index=0)

    # Category count summary
    st.sidebar.markdown("### Category Counts")
    counts = filtered_df["predict"].value_counts()
    for cat, count in counts.items():
        st.sidebar.write(f"{cat}: {count}")

    # Prepare embeddings and dimensionality reduction
    embeddings = np.array(filtered_df["embedding"].tolist())

    if embeddings.shape[0] < 2:
        st.warning("Not enough data to generate UMAP projection. Try selecting more categories.")
        st.stop()


    @st.cache_data
    def run_umap(embeds, n_components):
        n_neighbors = min(15, max(2, embeds.shape[0] - 1))
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", n_components=n_components)
        return reducer.fit_transform(embeds)


    n_dims = 3 if dimensionality == "3D" else 2
    coords = run_umap(embeddings, n_components=n_dims)

    if n_dims == 3:
        filtered_df["x"] = coords[:, 0]
        filtered_df["y"] = coords[:, 1]
        filtered_df["z"] = coords[:, 2]
    else:
        filtered_df["x"] = coords[:, 0]
        filtered_df["y"] = coords[:, 1]

    # Clustering method selection and parameters
    clustering_method = st.sidebar.selectbox("Clustering Method", ["DBSCAN", "HDBSCAN"])

    if clustering_method == "DBSCAN":
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5, step=0.1)
        min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    elif clustering_method == "HDBSCAN":
        min_cluster_size = st.sidebar.slider("HDBSCAN min_cluster_size", 3, 50, 5)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    else:
        clusterer = None

    clusters = clusterer.fit_predict(coords)
    filtered_df["cluster"] = clusters

    # Color column toggle
    color_col = "predict" if color_mode == "Category" else "cluster"
    label_encoder = LabelEncoder()
    filtered_df["color_code"] = label_encoder.fit_transform(filtered_df[color_col].astype(str))

    # Plot setup
    if n_dims == 3:
        fig = go.Figure()
        for value in filtered_df[color_col].unique():
            group_df = filtered_df[filtered_df[color_col] == value]
            fig.add_trace(go.Scatter3d(
                x=group_df["x"],
                y=group_df["y"],
                z=group_df["z"],
                mode="markers",
                name=str(value),
                text=group_df["title"],
                hovertemplate="<b>%{text}</b><br>Confidence: %{customdata[0]:.2f}<br>Score: %{customdata[1]}<extra></extra>",
                customdata=np.stack([group_df["confidence"], group_df["score"]], axis=-1),
                marker=dict(size=3, opacity=0.7)
            ))
        fig.update_layout(scene=dict(xaxis_title="UMAP X", yaxis_title="UMAP Y", zaxis_title="UMAP Z"))
    else:
        fig = go.Figure()
        for value in filtered_df[color_col].unique():
            group_df = filtered_df[filtered_df[color_col] == value]
            fig.add_trace(go.Scattergl(
                x=group_df["x"],
                y=group_df["y"],
                mode="markers",
                name=str(value),
                text=group_df["title"],
                hovertemplate="<b>%{text}</b><br>Confidence: %{customdata[0]:.2f}<br>Score: %{customdata[1]}<extra></extra>",
                customdata=np.stack([group_df["confidence"], group_df["score"]], axis=-1),
                marker=dict(size=6, opacity=0.7)
            ))
        # Draw convex hulls only in 2D
        for cluster_id in sorted(filtered_df["cluster"].unique()):
            if cluster_id == -1:
                continue
            cluster_df = filtered_df[filtered_df["cluster"] == cluster_id]
            if cluster_df.shape[0] < 3:
                continue
            try:
                points = cluster_df[["x", "y"]].values
                hull = ConvexHull(points)
                hull_pts = points[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                fig.add_trace(go.Scatter(
                    x=hull_pts[:, 0],
                    y=hull_pts[:, 1],
                    mode="lines",
                    line=dict(color="white", dash="dot"),
                    showlegend=False
                ))
            except Exception:
                continue
        fig.update_layout(
            title="Semantic Clusters with Category or Cluster Labels and Convex Hulls",
            xaxis_title="UMAP X",
            yaxis_title="UMAP Y",
            legend_title=color_col,
            width=1100,
            height=700,
            margin=dict(l=20, r=20, t=50, b=20)
        )

    # Render chart
    st.plotly_chart(fig, use_container_width=True, key="main_chart")

    # Preview table of filtered data
    with st.expander("🔎 View Filtered Data Table"):
        st.dataframe(filtered_df[["title", "url", "predict", "confidence", "score"]].reset_index(drop=True))

    # Click handler for data table
    st.markdown("## 📋 Selected Post Info")
    selected_title = st.text_input("Type or paste a keyword from the title to lookup:")

    if selected_title:
        match_df = df[df["title"].str.contains(selected_title, case=False, na=False)]
        if not match_df.empty:
            st.dataframe(match_df[["title", "url", "content", "predict", "confidence", "score"]].reset_index(drop=True))
        else:
            st.warning("No matching title found.")

    st.markdown("---")
    st.caption("Visualized using sentence-transformers + UMAP + DBSCAN/HDBSCAN | Click or search a point to inspect")