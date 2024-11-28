# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Charger le jeu de données
url = 'Financial_inclusion_dataset.csv'
df = pd.read_csv(url)

# Définir la variable cible et les caractéristiques
y = df['bank_account']
X = df.drop(columns=['bank_account', 'uniqueid', 'year'])

# Séparer les colonnes numériques et catégorielles
numeric_features = ['household_size', 'age_of_respondent']
categorical_features = [col for col in X.columns if col not in numeric_features]

# Préparer les transformateurs pour les pipelines
numeric_transformer = 'passthrough'
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Créer le préprocesseur avec ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Créer le pipeline complet avec préprocesseur et classificateur
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle entraîné
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
