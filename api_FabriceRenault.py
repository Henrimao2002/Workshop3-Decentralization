import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, request, jsonify

iris = load_iris()

MODEL_PATH = "neural_network_iris.h5"

# Création du modèle si non sauvegardé
if os.path.exists(MODEL_PATH):
    # Charger le modèle
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle chargé depuis le fichier existant.")
else:
    ValueError

# Création de l'API Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "API de prédiction pour le dataset Iris avec un Neural Network"

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Récupérer les paramètres de la requête
        sepal_length = float(request.args.get("sepal_length"))
        sepal_width = float(request.args.get("sepal_width"))
        petal_length = float(request.args.get("petal_length"))
        petal_width = float(request.args.get("petal_width"))
        
        # Préparer les données pour la prédiction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Prédire la classe
        predictions = model.predict(input_data)
        predicted_class = iris.target_names[np.argmax(predictions[0])]
        
        # Réponse standardisée
        response = {
            "confidence": float(np.max(predictions[0])),
            "prediction": predicted_class,
            "probabilities": predictions[0].tolist()  
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)