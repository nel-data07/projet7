import pandas as pd

file_path = "/Users/Nelly/Desktop/projet7/backend/clients_data.csv"  # Chemin de ton fichier CSV
df = pd.read_csv(file_path)

# Vérifier les colonnes présentes
print("Colonnes du fichier CSV :", df.columns)

# Vérifier les premières lignes du fichier
print("Aperçu des données :")
print(df.head())

