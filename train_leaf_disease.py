import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import logging

# Configure logging for feedback processing
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the classes (must match app.py)
CLASSES = {
    0: {'name': 'Healthy'},
    1: {'name': 'Powdery Mildew'},
    2: {'name': 'Black Spot'},
    3: {'name': 'Bacterial Leaf Spot'},
    4: {'name': 'Rust'},
    5: {'name': 'Leaf Blight'},
    6: {'name': 'Anthracnose'},
    7: {'name': 'Downy Mildew'},
    8: {'name': 'Leaf Curl'},
    9: {'name': 'Leaf Scorch'}
}

# Reverse mapping for training (name to index)
CLASS_TO_IDX = {data['name']: idx for idx, data in CLASSES.items()}

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
os.makedirs('feedback_images', exist_ok=True)

# Define a simple model (must match app.py's LeafDiseaseModel structure)
class SimpleLeafDiseaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleLeafDiseaseModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class FeedbackDataset(Dataset):
    def __init__(self, log_file, transform=None):
        self.transform = transform
        self.data = []
        self._load_feedback(log_file)

    def _load_feedback(self, log_file):
        if not os.path.exists(log_file):
            logging.warning(f"Feedback log file not found: {log_file}")
            return
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    # Remove timestamp and INFO prefix
                    feedback_data_str = line.split(' - Feedback - ', 1)[1].strip()
                    
                    # Parse key-value pairs
                    feedback_kv_pairs = {}
                    parts = feedback_data_str.split(', ')
                    for part in parts:
                        if ': ' in part:
                            key, value = part.split(': ', 1)
                            feedback_kv_pairs[key.strip()] = value.strip()

                    image_id = feedback_kv_pairs.get('Image ID')
                    predicted_label = feedback_kv_pairs.get('Predicted')
                    correct_label = feedback_kv_pairs.get('Correct')
                    image_path = feedback_kv_pairs.get('Image Path')

                    if not (image_id and correct_label):
                        logging.warning(f"Skipping feedback: Missing Image ID or Correct label - {line.strip()}")
                        continue

                    # Reconstruct image path if not explicitly logged (for older entries)
                    if image_path is None:
                        image_path = os.path.join('feedback_images', f"{image_id}.png")
                        logging.info(f"Reconstructed image path: {image_path}") # Log reconstruction

                    if os.path.exists(image_path) and correct_label in CLASS_TO_IDX:
                        self.data.append({'image_path': image_path, 'label': CLASS_TO_IDX[correct_label]})
                    else:
                        logging.warning(f"Skipping feedback: Image file not found or label invalid - Image Path: {image_path}, Correct Label: {correct_label}")
                except Exception as e:
                    logging.error(f"Error parsing feedback line '{line.strip()}': {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)
            
        return image, label

def fine_tune_model_with_feedback(log_file='feedback.log', model_path='models/leaf_disease_model.pth', num_epochs=5, learning_rate=0.0001):
    logging.info("Starting model fine-tuning with feedback...")
    
    # Image transformations (must match preprocess_image in app.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load feedback data
    feedback_dataset = FeedbackDataset(log_file, transform=transform)
    if len(feedback_dataset) == 0:
        logging.warning("No valid feedback data found to fine-tune the model.")
        return

    feedback_dataloader = DataLoader(feedback_dataset, batch_size=4, shuffle=True)

    # Load the existing model
    model = SimpleLeafDiseaseModel(num_classes=10)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            logging.info(f"Successfully loaded existing model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading existing model: {e}. Training a new model instead.")
    else:
        logging.warning(f"Existing model not found at {model_path}. Training a new model from scratch.")

    # Set model to training mode
    model.train()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Fine-tuning loop
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in feedback_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(feedback_dataloader):.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), model_path)
    logging.info(f"Fine-tuned model saved successfully to {model_path}!")

if __name__ == '__main__':
    # Print class_to_idx mapping for verification
    data_dir = 'data'  # Adjust if your dataset is in a subfolder
    if os.path.exists(data_dir):
        dataset = datasets.ImageFolder(data_dir)
        print('class_to_idx mapping:', dataset.class_to_idx)
    else:
        print('Data directory not found:', data_dir)
    # Uncomment the following to train or fine-tune
    # create_simple_model()
    fine_tune_model_with_feedback() 