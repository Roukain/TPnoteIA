import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm

# --- 1. Chemins et chargements ---
IMDB_ROOT = os.path.join(os.getcwd(), 'imdb', 'plain_text')

# Chargement des données supervisées (CSV)
train_df = pd.read_csv('imdb_train.csv')
test_df  = pd.read_csv('imdb_test.csv')

# Chargement des données non supervisées (parquet)
unsup_df = pd.read_parquet(os.path.join(IMDB_ROOT, 'unsupervised-00000-of-00001.parquet'))

# Textes et labels
X_train_text = train_df['text']
y_train      = train_df['label']
X_test_text  = test_df['text']
y_test       = test_df['label']
X_unsup_text = unsup_df['text']

# --- 2. Vectorisation TF-IDF ---
vectorizer = TfidfVectorizer(max_features=5000)
# On ajuste sur tous les textes pour inclure le vocabulaire non supervisé
vectorizer.fit(pd.concat([X_train_text, X_unsup_text]))

X_train = vectorizer.transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)
X_unsup = vectorizer.transform(X_unsup_text)

# Conversion en tenseurs PyTorch
X_train_t = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)
X_test_t  = torch.tensor(X_test.toarray(),  dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.long)
X_unsup_t = torch.tensor(X_unsup.toarray(), dtype=torch.float32)

# DataLoader pour supervisé
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# DataLoader pour unsupervisé (pas de labels)
unsup_loader = DataLoader(X_unsup_t, batch_size=32, shuffle=False)

# --- 3. Définition du modèle ---
class MLP_CE(nn.Module):
    def __init__(self, input_dim=5000, h1=128, h2=64, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.dropout = nn.Dropout(p)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# --- 4. Entraînement supervisé initial ---
def train_supervised(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for Xb, yb in tqdm.tqdm(loader, desc=f"Sup train epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

#  --- 5. Pseudo-étiquetage (self-training) ---
def pseudo_label(model, loader, threshold=0.9):
    model.eval()
    pseudo_texts = []
    pseudo_labels = []
    with torch.no_grad():
        for Xb in tqdm.tqdm(loader, desc="Génération pseudo-labels"):
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            mask = confs >= threshold
            if mask.sum() > 0:
                pseudo_texts.append(Xb[mask].cpu())
                pseudo_labels.append(preds[mask].cpu())
    if pseudo_texts:
        Xp = torch.cat(pseudo_texts, dim=0)
        yp = torch.cat(pseudo_labels, dim=0)
        print(f"{len(yp)} samples pseudo-étiquetés (seuil {threshold})")
        return Xp, yp
    else:
        print("Aucun pseudo-label généré, ajustez le seuil.")
        return None, None

# --- 6. Pipeline complet ---
if __name__ == '__main__':
    # 6.1 Entraînement initial
    model = MLP_CE()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    train_supervised(model, train_loader, ce, opt, epochs=5)

    # 6.2 Evaluation sur test (baseline)
    model.eval()
    with torch.no_grad():
        out_test = model(X_test_t)  # Ne pas passer y_test_t ici
        _, pred_test = out_test.max(1)

    print("\n=== Baseline (supervisé seul) ===")
    print(classification_report(y_test_t.cpu().numpy(), pred_test.cpu().numpy()))

    # 6.3 Pseudo-étiquetage du non supervisé
    Xp, yp = pseudo_label(model, unsup_loader, threshold=0.9)
    if Xp is not None:
        # 6.4 Création du DataLoader combiné
        sup_ds    = TensorDataset(X_train_t, y_train_t)
        pseudo_ds = TensorDataset(Xp, yp)
        comb_loader = DataLoader(ConcatDataset([sup_ds, pseudo_ds]), batch_size=32, shuffle=True)
        
        # 6.5 Ré-entraînement sur ensemble combiné
        model2 = MLP_CE()
        opt2 = optim.Adam(model2.parameters(), lr=1e-3)
        train_supervised(model2, comb_loader, ce, opt2, epochs=5)

        # 6.6 Evaluation semi-supervisé
        model2.eval()
        with torch.no_grad():
            out_test2 = model2(X_test_t)
            _, pred_test2 = out_test2.max(1)
        print("\n=== Après apprentissage semi-supervisé ===")
        print(classification_report(y_test_t.cpu().numpy(), pred_test2.cpu().numpy()))
