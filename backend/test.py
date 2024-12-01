if __name__ == "__main__":
    print("Premières lignes de clients_data :")
    print(clients_data.head())
    print("Colonne SK_ID_CURR existe : ", "SK_ID_CURR" in clients_data.columns)
    print("Valeurs nulles dans SK_ID_CURR : ", clients_data["SK_ID_CURR"].isnull().sum())
    print("Valeur maximale dans SK_ID_CURR : ", clients_data["SK_ID_CURR"].max())
