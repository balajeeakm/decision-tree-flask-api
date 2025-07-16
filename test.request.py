import requests

url = 'http://127.0.0.1:5000/predict'

# Sample input based on your model: 8 features
data = {
    'features': [0.5, 0.7, 4, 200, 3, 1, 0, 2]  # update as per your model's input
}

response = requests.post(url, json=data)
print("Prediction:", response.json())
