# Python-CLIP-Cluster

This Python script uses the CLIP model (by OpenAI) to extract high-dimensional image embeddings, clusters them using DBSCAN, and organizes the images into folders based on their cluster assignments. The result is a directory structure that groups visually similar images together.

## 🧠 Overview

- Uses the pre-trained `clip-vit-base-patch32` model from HuggingFace Transformers
- Extracts image embeddings using CLIP
- Applies dimensionality normalization via `StandardScaler`
- Performs unsupervised clustering using `DBSCAN`
- Separates outliers from valid clusters
- Automatically organizes the images into corresponding folders

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- transformers (`pip install transformers`)
- scikit-learn
- Pillow
- numpy

You can install dependencies using:

```bash
pip install torch torchvision transformers scikit-learn pillow numpy



