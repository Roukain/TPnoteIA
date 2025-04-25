# Question 4 : Implémentation d'un MLP avec PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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


# Convertir les données en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Créer un DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Définir le modèle MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(5000, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(criterion = nn.CrossEntropyLoss()):
    model = MLP()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraînement du modèle
    for epoch in tqdm.tqdm(range(10)):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
           # Compléter les étapes d'entrainement

    # Évaluation du modèle
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred_mlp = torch.max(outputs, 1)
        print("MLP :")
        print(classification_report(y_test_tensor.numpy(), y_pred_mlp.numpy()))

train()