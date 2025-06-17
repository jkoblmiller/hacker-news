# Hacker News Story Tagging & Clustering

This script builds a TF-IDF tagging and clustering model using Hacker News data.

## Features

- Merges story titles, content, and all related comments.
- Extracts top keywords using TF-IDF.
- Clusters stories using KMeans.

## Setup Instructions

1. Make sure you are using Python 3 (not Python 2.7).
2. Create and activate a virtual environment:

```bash
python3 -m venv hn_env
source hn_env/bin/activate  # On macOS/Linux

