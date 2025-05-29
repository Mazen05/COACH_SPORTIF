import argparse
import pandas as pd
import joblib
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Parser des arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--model_output', type=str, required=True)
parser.add_argument('--metrics_output', type=str, required=True)
args = parser.parse_args()

# Chargement des données
data = pd.read_csv(args.data_path)

X = data.drop("performance", axis=1)
y = data["performance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Création des dossiers si nécessaire
os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)

# Sauvegarde du modèle
joblib.dump(model, args.model_output)
print(f"✅ Modèle sauvegardé sous {args.model_output}")

# Sauvegarde des métriques
metrics = {"mean_squared_error": mse}
with open(args.metrics_output, "w") as f:
    json.dump(metrics, f)
print(f"✅ Métriques sauvegardées sous {args.metrics_output}")
