import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tqdm

# Charger le dataset IMDb
df1 = pd.read_csv('imdb_train.csv')  
X_train = df1['text']
y_train = df1['label']

df2 = pd.read_csv('imdb_test.csv')  
X_test = df2['text']
y_test = df2['label']

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.fit_transform(X_test)

# Conversion en tenseurs PyTorch pour BCEWithLogitsLoss
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test.toarray(),  dtype=torch.float32)

# Les labels doivent être float et avoir une dimension en plus pour coller aux logits [N,1]
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)


# Créer un DataLoader
dataset = TensorDataset(X_train_tensor, y_train_tensor)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

# Définir le modèle MLP
class MLP_BCE(nn.Module):
    def __init__(self):
        super(MLP_BCE, self).__init__()
        self.fc1 = nn.Linear(5000, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

counts = np.bincount(y_train.astype(int))
pos_weight = torch.tensor([counts[0]/counts[1]], dtype=torch.float32)

def train_bce(model, data_loader, criterion, optimizer, epochs=10):
    model.train()
    # Entraînement du modèle
    for epoch in range(epochs):
        total_loss = 0.0
        for Xb, yb in tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(data_loader):.4f}")

    # Évaluation du modèle
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int().squeeze()
    print("\n=== Résultats avec BCEWithLogitsLoss pondérée ===")
    print(classification_report(y_test_tensor.cpu().numpy().astype(int), preds.cpu().numpy()))

if __name__ == '__main__':
    # Initialisation
    torch.manual_seed(42)
    model = MLP_BCE()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_bce(model, loader, criterion, optimizer, epochs=10)