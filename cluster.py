import os
import shutil
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from transformers import CLIPProcessor, CLIPModel

# Ustawienia urządzenia (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Załaduj model CLIP i procesor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Funkcja do wyciągania cech z obrazu
def extract_features(image_path):
    """Extract image features using CLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    emb = image_features.cpu().numpy()[0]
    return emb

# Ścieżki folderów
input_folder = "Final_images_dataset"
output_folder = "Sorted_Images2"

# Upewnij się, że folder output istnieje
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lista ścieżek do obrazów
image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.lower().endswith((".png", ".jpg", ".jpeg"))]

# Wyciągnij cechy dla wszystkich obrazów
features = []
for image_path in image_paths:
    features.append(extract_features(image_path))
features = np.array(features)

# Normalizacja cech
features = StandardScaler().fit_transform(features)

# Klasteryzacja za pomocą DBSCAN z dostosowanymi parametrami
clustering = DBSCAN(eps=28.11, min_samples=3, metric='euclidean').fit(features)
labels = clustering.labels_

# Sprawdzenie poprawności klastrów
valid_clusters = []
for cluster_id in set(labels):
    if cluster_id != -1:  # Ignoruj outliery
        # Weryfikacja: sprawdzenie, czy klaster ma wystarczającą liczbę obrazów
        cluster_images = [img for img, lbl in zip(image_paths, labels) if lbl == cluster_id]
        if len(cluster_images) >= 2:  # Można zmienić minimalną liczbę obrazów w klastrze
            valid_clusters.append(cluster_id)

# Znajdź liczbę klastrów (bez outlierów oznaczonych -1)
num_clusters = len(valid_clusters)
print(f"Liczba klastrów (poprawnych): {num_clusters}")

# Tworzenie folderów dla klastrów i przenoszenie obrazów
for cluster_id in valid_clusters:
    cluster_folder = os.path.join(output_folder, f"cluster_{cluster_id}")
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)

    for img_path, lbl in zip(image_paths, labels):
        if lbl == cluster_id:
            shutil.copy(img_path, cluster_folder)

# Tworzenie folderu dla outlierów
outliers_folder = os.path.join(output_folder, "outliers")
if not os.path.exists(outliers_folder):
    os.makedirs(outliers_folder)

# Przenoszenie obrazów do folderu "outliers", jeśli są oznaczone jako outliery
for img_path, lbl in zip(image_paths, labels):
    if lbl == -1:
        shutil.copy(img_path, outliers_folder)

print("Obrazy zostały podzielone na klastry i przeniesione do odpowiednich folderów.")

