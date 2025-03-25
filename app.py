import os
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
import timm
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=5)
model.load_state_dict(torch.load("models/alzheimers_model1.pth"))
model = model.to(device)
model.eval()

# Define the classes for Alzheimer's stages
class_names = ['AD', 'CN', 'MCI', 'EMCI', 'LMCI']

# Data transformation for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Route for the homepage
@app.route('/')
def home():
    return render_template('index3.html')

# Route for classifying the image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the image
        img = Image.open(file.stream)

        # Convert to RGB if the image is grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply transformations
        img = transform(img).unsqueeze(0).to(device)

        # Make the prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)

import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
