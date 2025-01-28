import requests
import numpy as np
from sklearn.datasets import load_iris
import random

iris = load_iris()  # Call the function to load the dataset, not just reference it.

# Plages des caractéristiques pour chaque type de fleur
SEPAL_LENGTH_RANGE = (4.3, 7.9)
SEPAL_WIDTH_RANGE = (2.0, 4.4)
PETAL_LENGTH_RANGE = (1.0, 6.9)
PETAL_WIDTH_RANGE = (0.1, 2.5)

# Générer une fleur aléatoire
def generate_random_flower():
    sepal_length = round(random.uniform(*SEPAL_LENGTH_RANGE), 2)
    sepal_width = round(random.uniform(*SEPAL_WIDTH_RANGE), 2)
    petal_length = round(random.uniform(*PETAL_LENGTH_RANGE), 2)
    petal_width = round(random.uniform(*PETAL_WIDTH_RANGE), 2)
    
    # Calculate the closest match for the generated flower
    distances = np.linalg.norm(iris.data - np.array([sepal_length, sepal_width, petal_length, petal_width]), axis=1)
    closest_idx = np.argmin(distances)
    actual_class = iris.target[closest_idx]
    
    return {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
        "actual_class": actual_class  # Store the actual class for accuracy calculation
    }

# URLs des modèles des membres
MODEL_URLS = [
    "https://072d-89-30-29-68.ngrok-free.app/predict",  # Modèle 1
    "https://9f58-89-30-29-68.ngrok-free.app/predict",   # Modèle 3
    "https://e064-89-30-29-68.ngrok-free.app/predict"  # Modèle 3
]

# Fonction pour récupérer les prédictions
def get_predictions(input_data):
    predictions = []
    for url in MODEL_URLS:
        try:
            response = requests.get(url, params=input_data)
            if response.status_code == 200:
                prediction_data = response.json()
                predictions.append(prediction_data["probabilities"])  # Récupérer les probabilités
            else:
                print(f"Erreur avec le modèle à {url}: {response.status_code}")
        except Exception as e:
            print(f"Impossible de contacter {url}: {e}")
    return predictions

# Fonction de consensus
def consensus_prediction(predictions):
    # Convertir les prédictions en numpy array
    predictions = np.array(predictions, dtype=float)
    
    # Calculer la moyenne des probabilités (consensus)
    mean_prediction = np.mean(predictions, axis=0)
    
    # Classe prédite (index avec la probabilité maximale)
    predicted_class = np.argmax(mean_prediction)
    confidence = np.max(mean_prediction)
    
    return predicted_class, confidence, mean_prediction

# Exemple : Sélectionner 5 fleurs aléatoires dans le dataset Iris
num_predictions = 5
model_accuracies = [1, 1, 1]  # Start precision for all models at 1
for sample_idx in range(num_predictions):
    random_flower = generate_random_flower()  # Générer une fleur aléatoire
    print(f"Fleur {random_flower}")
    
    # Convertir l'entrée en dictionnaire pour l'API
    input_data = {
        "sepal_length": random_flower["sepal_length"],
        "sepal_width": random_flower["sepal_width"],
        "petal_length": random_flower["petal_length"],
        "petal_width": random_flower["petal_width"]
    }
    
    print(f"\nPrédictions pour l'échantillon {sample_idx}")
    
    # Récupérer les prédictions de chaque modèle
    predictions = get_predictions(input_data)
    
    if predictions:
        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction)
            actual_class = random_flower["actual_class"]
            correct = predicted_class == actual_class
            
            # Adjust precision based on whether the model was correct or not
            if correct:
                model_accuracies[i] = model_accuracies[i]  # Keep precision as 1 if correct
            else:
                model_accuracies[i] -= 0.5  # Decrease precision by 0.5 if incorrect

            # Print model results with dynamic precision
            print(f"Modèle {i+1}: {iris.target_names[predicted_class]}, Confiance: {np.max(prediction):.4f}, Précision: {model_accuracies[i]:.4f}")
        
        # Calculer le consensus
        predicted_class, confidence, mean_prediction = consensus_prediction(predictions)
        
        # Afficher le consensus
        print(f"\nConsensus : {iris.target_names[predicted_class]} (Confiance: {confidence:.4f})")
        print(f"Probabilités moyennes du consensus : {mean_prediction}")
    else:
        print("Aucune prédiction n'a pu être obtenue.")
