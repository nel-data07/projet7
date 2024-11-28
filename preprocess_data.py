import pandas as pd

# Charger les données
file_path = "/Users/Nelly/Desktop/projet 7/data/application_train.csv"
df = pd.read_csv(file_path)

# Vérifier les valeurs manquantes
print("Valeurs manquantes avant le traitement :")
print(df.isnull().sum())

# 1. Supprimer les lignes où CODE_GENDER est manquant
df = df.dropna(subset=["CODE_GENDER"])
print(f"Lignes supprimées pour 'CODE_GENDER' : {df.shape}")

# 2. Remplacer les valeurs manquantes par des statistiques pertinentes
# Remplacer les valeurs manquantes de AMT_ANNUITY par la médiane
df["AMT_ANNUITY"].fillna(df["AMT_ANNUITY"].median(), inplace=True)

# Remplacer les valeurs manquantes de AMT_GOODS_PRICE par la médiane
df["AMT_GOODS_PRICE"].fillna(df["AMT_GOODS_PRICE"].median(), inplace=True)

# 3. Convertir les colonnes en types numériques si nécessaire
# Convertir CODE_GENDER et FLAG_OWN_CAR en valeurs numériques
df["CODE_GENDER"] = df["CODE_GENDER"].apply(lambda x: 1 if x == "M" else 0)  # 1 = Male, 0 = Female
df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].apply(lambda x: 1 if x == "Y" else 0)  # 1 = Yes, 0 = No

# Vérifier les valeurs manquantes après le traitement
print("Valeurs manquantes après le traitement :")
print(df.isnull().sum())

# Enregistrer les données prétraitées
processed_file_path = "/Users/Nelly/Desktop/projet7/application_train_processed.csv"
df.to_csv(processed_file_path, index=False)
print(f"Données prétraitées enregistrées dans : {processed_file_path}")

