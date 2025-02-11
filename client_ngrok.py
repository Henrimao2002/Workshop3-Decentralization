import requests
import numpy as np
import json
import random
import os
from sklearn.datasets import load_iris

# Charger l'Iris dataset
iris = load_iris()

# Paths for the JSON database and model URLs
MODEL_BALANCES_FILE = "model_balances.json"

MODEL_URLS = [
    "https://072d-89-30-29-68.ngrok-free.app/predict",  # Model 1
    "https://9f58-89-30-29-68.ngrok-free.app/predict",   # Model 2
    "https://e064-89-30-29-68.ngrok-free.app/predict",   # Model 3
    "https://e69b-89-30-29-68.ngrok-free.app/predict"    # Model 4
]

# Default model balance (deposit)
DEFAULT_BALANCE = 1000.0

# Initialize model balances if the JSON file doesn't exist
def initialize_model_balances():
    if not os.path.exists(MODEL_BALANCES_FILE):
        balances = {
            "model_1": {"balance": DEFAULT_BALANCE, "accuracy": 0.0, "predictions": 0},
            "model_2": {"balance": DEFAULT_BALANCE, "accuracy": 0.0, "predictions": 0},
            "model_3": {"balance": DEFAULT_BALANCE, "accuracy": 0.0, "predictions": 0},
            "model_4": {"balance": DEFAULT_BALANCE, "accuracy": 0.0, "predictions": 0}
        }
        with open(MODEL_BALANCES_FILE, 'w') as file:
            json.dump(balances, file)

# Function to read model balances from JSON
def read_model_balances():
    with open(MODEL_BALANCES_FILE, 'r') as file:
        return json.load(file)

# Function to update model balances and accuracy
def update_model_balance(model_name, is_accurate):
    balances = read_model_balances()
    
    if model_name in balances:
        model = balances[model_name]
        # Adjust the balance based on prediction accuracy
        if not is_accurate:
            model["balance"] -= 50  # Deduct for bad predictions (e.g., 50 euros per bad prediction)
        
        # Update prediction accuracy
        model["predictions"] += 1
        model["accuracy"] = (model["accuracy"] * (model["predictions"] - 1) + (1 if is_accurate else 0)) / model["predictions"]
        
        with open(MODEL_BALANCES_FILE, 'w') as file:
            json.dump(balances, file)

# Function to get the model's weight based on accuracy
def get_model_weight(model_name):
    balances = read_model_balances()
    if model_name in balances:
        accuracy = balances[model_name]["accuracy"]
        # Map accuracy to weight (e.g., accuracy ranges from 0 to 1, weight from 0 to 1)
        return accuracy
    return 0.0  # Default to 0 if not found

# Function to calculate consensus prediction with weighted models
def weighted_consensus_prediction(predictions):
    # Assurez-vous que les prédictions sont sous forme de tableaux numériques
    weights = [get_model_weight(f"model_{i+1}") for i in range(len(predictions))]

    # Normaliser les poids pour qu'ils fassent la somme de 1
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1 / len(weights)] * len(weights)  # Poids uniformes si tous les modèles sont mauvais

    # Calcul de la somme pondérée des prédictions
    weighted_predictions = np.array([
        np.array(pred) * weight for pred, weight in zip(predictions, weights)
    ])
    
    mean_prediction = np.mean(weighted_predictions, axis=0)

    # Calcul de la classe prédite et de la confiance
    predicted_class = np.argmax(mean_prediction)
    confidence = np.max(mean_prediction)
    
    return predicted_class, confidence, mean_prediction


def get_predictions_from_models(features):
    predictions = []
    for url in MODEL_URLS:
        try:
            response = requests.get(url, params=features)
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                # Convertir la prédiction en classe numérique
                if prediction == "setosa":
                    predictions.append([1.0, 0.0, 0.0])  # Setosa
                elif prediction == "versicolor":
                    predictions.append([0.0, 1.0, 0.0])  # Versicolor
                elif prediction == "virginica":
                    predictions.append([0.0, 0.0, 1.0])  # Virginica
                else:
                    predictions.append([0.0, 0.0, 0.0])  # Cas où la prédiction est invalide
            else:
                print(f"Error in model prediction from {url}: {response.status_code}")
                predictions.append([1/3, 1/3, 1/3])  # En cas d'erreur, utiliser des valeurs neutres
        except Exception as e:
            print(f"Error connecting to {url}: {e}")
            predictions.append([1/3, 1/3, 1/3])  # En cas d'erreur, utiliser des valeurs neutres
    return predictions


# Simulating a prediction batch
def simulate_prediction_batch(round_num, true_label):
    print(f"\n--- Round {round_num} ---")
    
    # Simulating feature vector (e.g., iris features, 4 values)
    random_index = random.randint(0, len(iris.data) - 1)
    random_flower = iris.data[random_index]

    # Créer un dictionnaire avec les caractéristiques de l'échantillon choisi
    features = {
        "sepal_length": random_flower[0],  # sepal_length est la 1ère caractéristique
        "sepal_width": random_flower[1],   # sepal_width est la 2ème caractéristique
        "petal_length": random_flower[2],  # petal_length est la 3ème caractéristique
        "petal_width": random_flower[3]    # petal_width est la 4ème caractéristique
    }
    
    # Get real predictions from the models
    predictions = get_predictions_from_models(features)
    
    # Checking accuracy and updating balances
    for i, pred in enumerate(predictions):
        predicted_class = np.argmax(pred)
        is_accurate = predicted_class == true_label
        model_name = f"model_{i+1}"
        update_model_balance(model_name, is_accurate)
    
    # Calculate consensus with weights
    predicted_class, confidence, mean_prediction = weighted_consensus_prediction(predictions)
    print(f"Consensus predicted class: {predicted_class}, Confidence: {confidence}")
    print(f"Mean prediction probabilities: {mean_prediction}")
    
    # Print current model balances
    balances = read_model_balances()
    for model_name in balances:
        model = balances[model_name]
        print(f"{model_name} - Balance: {model['balance']:.2f}, Accuracy: {model['accuracy']:.4f}, Predictions: {model['predictions']}")

def run_simulation(num_rounds):
    initialize_model_balances()
    
    for round_num in range(1, num_rounds + 1):
        # Assuming true_label is 1 for simplicity (you can change this as needed)
        simulate_prediction_batch(round_num, true_label=1)

# Run the simulation for 10 rounds
run_simulation(5)