from flask import Flask, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pandas as pd
import numpy as np

# Load disease info dataset
disease_info = pd.read_csv('Model_assest/disease_info.csv')

# Device agnostic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = models.resnet18(pretrained=True)
num_classes = 38
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('Model_assest/model.pth', map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def prediction(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    output = model(image)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

# --- API Key setup ---
API_KEY = "HelloWorld123GDC"

def require_api_key(func):
    def wrapper(*args, **kwargs):
        key = request.headers.get('x-api-key')
        print("Received key:", key)  # <- debug line
        if key and key == API_KEY:
            return func(*args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized"}), 401
    wrapper.__name__ = func.__name__
    return wrapper

# --- API endpoint ---
@app.route('/predict', methods=['POST'])
@require_api_key
def api_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    filename = image.filename
    file_path = os.path.join('static/uploads', filename)
    os.makedirs('static/uploads', exist_ok=True)
    image.save(file_path)

    # Prediction
    pred = prediction(file_path)
    response = {
        'prediction': disease_info['disease_name'][pred],
        'description': disease_info['description'][pred],
        'possible_steps': disease_info['Possible Steps'][pred]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
