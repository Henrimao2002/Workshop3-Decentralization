import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, request, jsonify

# Charger le dataset Iris
iris = load_iris()
X = iris.data  # Caractéristiques
y = iris.target.reshape(-1, 1)  # Labels (reshape pour OneHotEncoder)

# Encodage One-Hot des classes de sortie
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

MODEL_PATH = "neural_network_iris.h5"

# Créer un réseau de neurones dense
model = Sequential([
    Dense(16, input_shape=(4,), activation='relu'),  # Couche cachée (16 neurones)
    Dense(3, activation='softmax')                  # Couche de sortie (3 neurones pour 3 classes)
])

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

# Sauvegarder le modèle
model.save(MODEL_PATH)
print(f"Modèle entraîné et sauvegardé dans {MODEL_PATH}")

# Évaluation du modèle
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Performance du modèle - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
