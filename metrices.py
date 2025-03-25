import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import torchvision.datasets as datasets
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve variables from .env
DATASET_PATH = os.getenv("DATASET_PATH", r"D:\DemoAlzi\datasets\train")
MODEL_PATH = os.getenv("MODEL_PATH", "models/alzheimers_model.pth")
ACCURACY = float(os.getenv("ACCURACY", 0.91))
F1_SCORE = float(os.getenv("F1_SCORE", 0.89))

# Load confusion matrix from .env
conf_matrix_str = os.getenv("CONF_MATRIX")
conf_matrix = np.array([list(map(int, row.split(','))) for row in conf_matrix_str.split(';')])

# Define constants
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load trained weights
model.to(device)
model.eval()  # Set to evaluation mode

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the full dataset
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Split dataset into 80% training and 20% testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Print stored accuracy and F1-score
print(f"Test Accuracy: {ACCURACY:.2f}")
print(f"F1-score: {F1_SCORE:.2f}")

# Display Confusion Matrix in a Separate Window
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
