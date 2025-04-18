import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm

# 1. Chargement et préparation des données
train_df = pd.read_csv('imdb_train.csv')
test_df  = pd.read_csv('imdb_test.csv')

X_train_text, y_train = train_df['text'], train_df['label']
X_test_text,  y_test  = test_df['text'],  test_df['label']

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

# Conversion en tenseurs PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test.toarray(),  dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.long).to(device)

# DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 2. Modèles
class MLP_CE(nn.Module):
    """
    MLP pour CrossEntropyLoss (sortie 2 neurones)
    """
    def __init__(self, input_dim=5000, hidden1=128, hidden2=64, dropout_p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class MLP_BCE(nn.Module):
    """
    MLP pour BCEWithLogitsLoss (sortie 1 neurone)
    """
    def __init__(self, input_dim=5000, hidden1=128, hidden2=64, dropout_p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# 3. Fonctions d'entraînement et d'évaluation

def train_CE(model, loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in tqdm.tqdm(loader, desc=f'Epoch CE {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'[CE] Epoch {epoch+1} loss: {epoch_loss/len(loader):.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
    print("\n=== Résultats avec CrossEntropyLoss ===")
    print(classification_report(y_test_tensor.cpu().numpy(), preds.cpu().numpy()))


def train_BCE(model, loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in tqdm.tqdm(loader, desc=f'Epoch BCE {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            labels_float = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels_float)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'[BCE] Epoch {epoch+1} loss: {epoch_loss/len(loader):.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int().squeeze()
    print("\n=== Résultats avec BCEWithLogitsLoss pondérée ===")
    print(classification_report(y_test_tensor.cpu().numpy(), preds.cpu().numpy()))


# 4. Exécution principale
if __name__ == '__main__':
    # version CrossEntropy
    model_ce = MLP_CE().to(device)
    optimizer_ce = optim.Adam(model_ce.parameters(), lr=0.001)
    criterion_ce = nn.CrossEntropyLoss()
    train_CE(model_ce, train_loader, criterion_ce, optimizer_ce, epochs=10)

    # version BCEWithLogitsLoss pondérée
    model_bce = MLP_BCE().to(device)
    optimizer_bce = optim.Adam(model_bce.parameters(), lr=0.001)
    pos_weight = torch.tensor([2.0], device=device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_BCE(model_bce, train_loader, criterion_bce, optimizer_bce, epochs=10)
