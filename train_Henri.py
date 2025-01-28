from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names  # ["setosa", "versicolor", "virginica"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
knn_model = KNeighborsClassifier().fit(X_train, y_train)

# Evaluate accuracy
print(f"KNN Accuracy: {accuracy_score(y_test, knn_model.predict(X_test))}")

# API Deployment
app = Flask(__name__)

# Standardized API for Predictions
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get the features from the query parameters
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        # Debugging: print the values received from the URL
        print(f"Received values: sepal_length={sepal_length}, sepal_width={sepal_width}, petal_length={petal_length}, petal_width={petal_width}")

        # Prepare the feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict with the selected model
        prediction = knn_model.predict(features)[0]
        prediction_probabilities = knn_model.predict_proba(features)[0]  # Get class probabilities

        # Convert the numeric prediction to a string label
        prediction_str = class_names[prediction]
        
        # Get the confidence as the probability of the predicted class
        confidence = float(prediction_probabilities[prediction])

        # Prepare the response with rounded probabilities to 3 decimal places
        response = {
            "prediction": prediction_str,
            "confidence": round(confidence, 3),
            "probabilities": [round(float(prob), 3) for prob in prediction_probabilities]  # Round probabilities to 3 decimal places
        }

        return jsonify(response)
    
    except Exception as e:
        # Catch any error and return it in the response
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003)
