import joblib
import os

# Chemin absolu vers le fichier
MODEL_PATH = "/Users/Nelly/Desktop/projet7/backend/best_model_lgb_bal.pkl"

try:
    print(f"Chargement du modèle depuis : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Modèle chargé avec succès.")
except FileNotFoundError:
    print(f"Fichier modèle introuvable : {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

