import os
import pandas as pd
import joblib
import lightgbm as lgb

# Chemin vers les données prétraitées
DATA_PATH = "/Users/Nelly/Desktop/projet 7/data/clients_data.csv"

# Vérifiez si le fichier de données existe
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Fichier de données introuvable : {DATA_PATH}")

# Charger les données prétraitées
print("Chargement des données prétraitées...")
df = pd.read_csv(DATA_PATH)

# Vérifier si 'SK_ID_CURR' et 'TARGET' sont présents
if 'SK_ID_CURR' not in df.columns or 'TARGET' not in df.columns:
    raise ValueError("Les colonnes 'SK_ID_CURR' et 'TARGET' doivent être présentes dans les données.")

# Nettoyer les noms des colonnes (supprimer les caractères spéciaux et espaces)
df.columns = df.columns.str.replace(r"[^\w\s]", "_", regex=True).str.replace(" ", "_")
print(f"Noms de colonnes nettoyés : {df.columns.tolist()}")

# Exclure 'SK_ID_CURR' et 'TARGET' lors de la définition des features
X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

# Vérifier les colonnes utilisées pour l'entraînement
print(f"Colonnes utilisées pour l'entraînement : {X.columns.tolist()}")

# Vérifier les valeurs manquantes
if X.isnull().sum().sum() > 0:
    raise ValueError("Des valeurs manquantes existent encore dans les données !")

# Entraîner le modèle
print("Entraînement du modèle LightGBM...")
model = lgb.LGBMClassifier(random_state=42)
model.fit(X, y)

# Sauvegarder le modèle entraîné
MODEL_PATH = os.path.join("backend", "best_model_lgb_no.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Modèle sauvegardé dans : {MODEL_PATH}")

# Sauvegarder les colonnes utilisées pour l'entraînement
FEATURES_PATH = os.path.join("backend", "selected_features.txt")
with open(FEATURES_PATH, "w") as f:
    f.write(",".join(X.columns.tolist()))
print(f"Colonnes sauvegardées dans : {FEATURES_PATH}")
