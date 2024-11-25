import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Charger les datasets
train_data = pd.read_csv("/Users/Nelly/Desktop/projet 7/data/application_train.csv")
test_data = pd.read_csv("/Users/Nelly/Desktop/projet 7/data/application_test.csv")

# Créer un rapport de Data Drift
data_drift_report = Report(metrics=[DataDriftPreset()])

# Sélectionner uniquement les colonnes nécessaires pour l'analyse
common_columns = list(set(train_data.columns).intersection(set(test_data.columns)))
train_data = train_data[common_columns]
test_data = test_data[common_columns]

# Générer le rapport
data_drift_report.run(reference_data=train_data, current_data=test_data)

# Exporter le rapport en HTML
data_drift_report.save_html("data_drift_report.html")

print("Le rapport de Data Drift a été généré : 'data_drift_report.html'")

