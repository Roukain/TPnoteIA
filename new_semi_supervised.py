# Importer les bibliothèques nécessaires
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np

# Chargement du dataset IMDb
dataset = load_dataset("imdb")

# Séparation des données en étiquetées et non étiquetées
labeled_data = dataset['train'].select(range(5000))  # 5000 étiquetées
unlabeled_data = dataset['train'].select(range(5000, len(dataset['train'])))  # Le reste non étiqueté

# Modèle pour la classification supervisée
classifier = pipeline("text-classification", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")

# Fonction pour obtenir les embeddings BERT pour le clustering
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Moyenne des tokens
    return embeddings

# Extraire les embeddings pour les données non étiquetées
texts = unlabeled_data['text']
embeddings = get_embeddings(texts)

# Appliquer KMeans pour le clustering (2 clusters : positif ou négatif)
kmeans = KMeans(n_clusters=2)
kmeans.fit(embeddings.numpy())
clusters = kmeans.predict(embeddings.numpy())

# Annoter les données non étiquetées avec les clusters
unlabeled_data = unlabeled_data.add_column('label', clusters)

# Vectorisation des textes étiquetés pour l'entraînement supervisé
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(labeled_data['text'])
y_train = labeled_data['label']  # Utiliser les labels réels dans le dataset étiqueté

# Entraînement du classifieur Logistic Regression
classifier_lr = LogisticRegression()
classifier_lr.fit(X_train, y_train)

# Prédictions sur les données non étiquetées
X_unlabeled = vectorizer.transform(unlabeled_data['text'])
y_unlabeled = classifier_lr.predict(X_unlabeled)

# Ajouter les étiquettes prédites aux données non étiquetées
unlabeled_data = unlabeled_data.add_column('predicted_label', y_unlabeled)

# Afficher quelques résultats
print("Exemple de données étiquetées :")
print(labeled_data[0])

print("\nExemple de données non étiquetées avec prédictions :")
print(unlabeled_data[0])

# Sauvegarder les données annotées (facultatif)
unlabeled_data.to_csv("annotated_imdb.csv")
