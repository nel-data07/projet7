import pkg_resources

packages = [
    "flask", "gunicorn", "lightgbm", "pandas", "numpy",
    "evidently", "pytest", "requests", "streamlit",
    "shap", "matplotlib", "scikit-learn", "flask-cors", "joblib","gdown"
]

for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"{pkg}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{pkg}: Non installé")

