from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load your trained model
model = joblib.load('decision_tree_model.pkl')

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Decision Tree Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Example input: [0.5, 0.7, 4, 200, 3, 1, 0, 2]
    input_features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(input_features)
    
    result = "LEFT" if prediction[0] == 1 else "NOT LEFT"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
