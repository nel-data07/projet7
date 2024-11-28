import pandas as pd
import lightgbm as lgb
import os

# Chemin vers les données prétraitées
DATA_PATH = os.path.join("application_train_processed.csv")

# Charger les données prétraitées
print("Chargement des données prétraitées...")
df = pd.read_csv(DATA_PATH)

# Définir les features et la cible
selected_features = ["CODE_GENDER", "FLAG_OWN_CAR", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                     "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
X = df[selected_features]
y = df["TARGET"]

# Vérification des colonnes manquantes après le prétraitement
if X.isnull().sum().sum() > 0:
    raise ValueError("Des valeurs manquantes existent encore dans les données !")

# Créer et entraîner le modèle
print("Entraînement du modèle...")
model = lgb.LGBMClassifier()
model.fit(X, y)

# Sauvegarder le modèle entraîné
print("Sauvegarde du modèle...")
MODEL_PATH = os.path.join("models", "lightgbm_model.txt")
os.makedirs("models", exist_ok=True)
model.booster_.save_model(MODEL_PATH)

# Sauvegarder les colonnes utilisées pour l'entraînement
print("Sauvegarde des colonnes sélectionnées...")
FEATURES_PATH = os.path.join("models", "selected_features.txt")
with open(FEATURES_PATH, "w") as f:
    f.write(",".join(selected_features))

print("Modèle et colonnes sauvegardés avec succès !")