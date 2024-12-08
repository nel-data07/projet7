import os
import pandas as pd

# Chemins des fichiers
input_file = "/Users/Nelly/Desktop/projet 7/clients_data.csv"  
output_file = "/Users/Nelly/Desktop/projet7/backend/clients_data.csv"  

# Taille maximale en mégaoctets
max_size_mb = 16  # Réduire à 16 Mo

# Charger le fichier d'entrée
print("Chargement du fichier d'origine...")
df = pd.read_csv(input_file)

# Calculer la taille d'une ligne moyenne
sample_size = 1000  # Nombre de lignes pour estimer la taille
sample = df.head(sample_size)
avg_line_size = sample.memory_usage(deep=True).sum() / len(sample)

# Calculer le nombre de lignes à garder pour ne pas dépasser max_size_mb
max_size_bytes = max_size_mb * 1024 * 1024
max_lines = int(max_size_bytes / avg_line_size)

# Réduire le DataFrame
print(f"Fichier original : {len(df)} lignes. Limite fixée : {max_lines} lignes.")
reduced_df = df.head(max_lines)

# Enregistrer le fichier réduit
print(f"Enregistrement du fichier réduit ({len(reduced_df)} lignes)...")
reduced_df.to_csv(output_file, index=False)

# Vérifier la taille du fichier réduit
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"Fichier réduit enregistré : {output_file} ({file_size_mb:.2f} Mo)")
