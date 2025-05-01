import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm
import numpy as np


# Nombre d'évaluations à réaliser
n_runs = 5
results = []

# 1. Chargement des données spam
df_spam = pd.read_csv("spam.csv", sep='\t', names=["label", "text"])
df_spam['label'] = df_spam['label'].map({'ham': 0, 'spam': 1})

# Découpage train/test pour le dataset spam (50% pour chaque)
# --- Réponse à la question 2 : Découpage des données en train et test pour le dataset spam
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(
    df_spam['text'], df_spam['label'], test_size=0.5, stratify=df_spam['label'], random_state=42
)

# 2. Vectorisation TF-IDF pour spam
vectorizer_spam = TfidfVectorizer(max_features=5000)
X_train_spam_tfidf = vectorizer_spam.fit_transform(X_train_spam)
X_test_spam_tfidf = vectorizer_spam.transform(X_test_spam)

# 3. Modèles classique pour spam (Arbre de décision, Random Forest, SVM)
print("=== Comparaison des modèles sur spam ===")

# Arbre de Décision
tree_model_spam = DecisionTreeClassifier(random_state=42)
tree_model_spam.fit(X_train_spam_tfidf, y_train_spam)
tree_preds_spam = tree_model_spam.predict(X_test_spam_tfidf)
# --- Réponse à la question 3 : Quel modèle donne les meilleurs résultats pour chaque dataset ?
print("=== Arbre de Décision (Spam) ===")
print(classification_report(y_test_spam, tree_preds_spam))

# Random Forest
forest_model_spam = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model_spam.fit(X_train_spam_tfidf, y_train_spam)
forest_preds_spam = forest_model_spam.predict(X_test_spam_tfidf)
print("=== Random Forest (Spam) ===")
print(classification_report(y_test_spam, forest_preds_spam))

# SVM Linéaire
svm_model_spam = LinearSVC()
svm_model_spam.fit(X_train_spam_tfidf, y_train_spam)
svm_preds_spam = svm_model_spam.predict(X_test_spam_tfidf)
print("=== SVM Linéaire (Spam) ===")
print(classification_report(y_test_spam, svm_preds_spam))

# 4. Implémentation MLP avec PyTorch pour spam
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(5000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Conversion en tenseurs
X_train_spam_tensor = torch.tensor(X_train_spam_tfidf.toarray(), dtype=torch.float32)
X_test_spam_tensor = torch.tensor(X_test_spam_tfidf.toarray(), dtype=torch.float32)
y_train_spam_tensor = torch.tensor(y_train_spam.values, dtype=torch.float32).unsqueeze(1)
y_test_spam_tensor = torch.tensor(y_test_spam.values, dtype=torch.float32).unsqueeze(1)

# Entraînement MLP
def train_mlp():
    model = MLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)

    # Dataset
    train_dataset = TensorDataset(X_train_spam_tensor, y_train_spam_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in tqdm.tqdm(range(20)):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Évaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_spam_tensor)
        predictions = (outputs > 0.5).float()
        print("=== MLP (Spam) ===")
        print(classification_report(y_test_spam_tensor.numpy(), predictions.numpy()))

train_mlp()

# 5. Chargement des données IMDb
df_train_imdb = pd.read_csv('imdb_train.csv')
df_test_imdb = pd.read_csv('imdb_test.csv')
X_train_imdb = df_train_imdb['text']
y_train_imdb = df_train_imdb['label']
X_test_imdb = df_test_imdb['text']
y_test_imdb = df_test_imdb['label']

# Vectorisation TF-IDF pour IMDb
vectorizer_imdb = TfidfVectorizer(max_features=5000)
X_train_imdb_tfidf = vectorizer_imdb.fit_transform(X_train_imdb)
X_test_imdb_tfidf = vectorizer_imdb.transform(X_test_imdb)

# Modèles classique pour IMDb (Arbre de décision, Random Forest, SVM)
print("=== Comparaison des modèles sur IMDb ===")

# Arbre de Décision
tree_model_imdb = DecisionTreeClassifier(random_state=42)
tree_model_imdb.fit(X_train_imdb_tfidf, y_train_imdb)
tree_preds_imdb = tree_model_imdb.predict(X_test_imdb_tfidf)
print("=== Arbre de Décision (IMDb) ===")
print(classification_report(y_test_imdb, tree_preds_imdb))

# Random Forest
forest_model_imdb = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model_imdb.fit(X_train_imdb_tfidf, y_train_imdb)
forest_preds_imdb = forest_model_imdb.predict(X_test_imdb_tfidf)
print("=== Random Forest (IMDb) ===")
print(classification_report(y_test_imdb, forest_preds_imdb))

# SVM Linéaire
svm_model_imdb = LinearSVC()
svm_model_imdb.fit(X_train_imdb_tfidf, y_train_imdb)
svm_preds_imdb = svm_model_imdb.predict(X_test_imdb_tfidf)
print("=== SVM Linéaire (IMDb) ===")
print(classification_report(y_test_imdb, svm_preds_imdb))

# Exécuter l'évaluation plusieurs fois
for i in range(n_runs):
    print(f"\n=== Evaluation {i+1} ===")

    # Découpage aléatoire des données (différentes partitions à chaque exécution)
    X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(
        df_spam['text'], df_spam['label'], test_size=0.5, stratify=df_spam['label'], random_state=np.random.randint(0, 10000)
    )

    # Vectorisation TF-IDF
    vectorizer_spam = TfidfVectorizer(max_features=5000)
    X_train_spam_tfidf = vectorizer_spam.fit_transform(X_train_spam)
    X_test_spam_tfidf = vectorizer_spam.transform(X_test_spam)

    # Entraînement et évaluation du modèle (par exemple, SVM Linéaire)
    svm_model_spam = LinearSVC()
    svm_model_spam.fit(X_train_spam_tfidf, y_train_spam)
    svm_preds_spam = svm_model_spam.predict(X_test_spam_tfidf)
    
    # Calculer le rapport de classification
    result = classification_report(y_test_spam, svm_preds_spam, output_dict=True)
    results.append(result)

    # Affichage de la métrique pour cette évaluation
    print(classification_report(y_test_spam, svm_preds_spam))

# Calculer les scores moyens et les variances pour chaque métrique
results_df = pd.DataFrame(results)
mean_scores = results_df.mean()
std_scores = results_df.std()

print("\n=== Scores moyens et écart-types après plusieurs évaluations ===")
print("Moyenne des scores :")
print(mean_scores)
print("Écart-type des scores :")
print(std_scores)