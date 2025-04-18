import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. Chargement des données IMDb ---
train_imdb = pd.read_csv('imdb_train.csv')
test_imdb  = pd.read_csv('imdb_test.csv')
X_train_i, y_train_i = train_imdb['text'], train_imdb['label']
X_test_i,  y_test_i  = test_imdb['text'],  test_imdb['label']

# --- 2. Chargement du dataset Spam ---
# Assure-toi d'avoir spam.csv à la racine du projet
spam = pd.read_csv('spam.csv')
if {'text','label'}.issubset(spam.columns):
    X_spam = spam['text']
    y_spam = spam['label']
else:
    # Ajuste selon la structure de ton spam.csv (ex. première colonne label, 2ᵉ text)
    X_spam = spam.iloc[:, 1]
    y_spam = spam.iloc[:, 0]
# Découpage train/test pour Spam (80/20)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_spam, y_spam, test_size=0.2, random_state=42
)

# --- 3. Définition des modèles à comparer ---
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'MultinomialNB':      MultinomialNB(),
    'SVM':                SVC(),
    'RandomForest':       RandomForestClassifier(n_estimators=100)
}

# --- 4. Évaluation sur IMDb ---
print("=== Comparaison sur IMDb ===")
for name, clf in models.items():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf',   clf)
    ])
    # Entraînement et test
    pipe.fit(X_train_i, y_train_i)
    preds = pipe.predict(X_test_i)
    print(f"\n-- {name} --")
    print(classification_report(y_test_i, preds))
    # Significativité via Cross-Validation 5-fold sur l'ensemble complet
    X_all = pd.concat([X_train_i, X_test_i])
    y_all = pd.concat([y_train_i, y_test_i])
    scores = cross_val_score(pipe, X_all, y_all, cv=5, scoring='accuracy')
    print(f"CV accuracy : mean = {scores.mean():.3f}, std = {scores.std():.3f}")

# --- 5. Évaluation sur le dataset Spam ---
print("\n=== Comparaison sur Spam ===")
for name, clf in models.items():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf',   clf)
    ])
    pipe.fit(X_train_s, y_train_s)
    preds = pipe.predict(X_test_s)
    print(f"\n-- {name} --")
    print(classification_report(y_test_s, preds))
    # CV pour significativité
    scores = cross_val_score(pipe, X_spam, y_spam, cv=5, scoring='accuracy')
    print(f"CV accuracy : mean = {scores.mean():.3f}, std = {scores.std():.3f}")
