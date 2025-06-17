import streamlit as st
import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image, ImageStat
import requests
from io import BytesIO
import cv2
import torch.nn as nn
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
import os
import logging
import hashlib

# Configure logging for feedback
logging.basicConfig(filename='feedback.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the classes for plant diseases
CLASSES = {
    0: {
        'name': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'symptoms': 'No visible symptoms. Leaf has uniform green color and normal texture.',
        'causes': 'Plant is well-maintained with proper nutrition and care.',
        'treatment': [
            'Continue regular maintenance',
            'Maintain proper watering schedule',
            'Ensure adequate sunlight',
            'Monitor for any changes'
        ],
        'prevention': [
            'Regular inspection of plants',
            'Proper spacing between plants',
            'Good air circulation',
            'Balanced fertilization'
        ]
    },
    1: {
        'name': 'Powdery Mildew',
        'description': 'A fungal disease that appears as white powdery spots on leaves.',
        'symptoms': 'White powdery coating on leaves, typically starting as small spots that expand.',
        'causes': 'Fungal infection (Erysiphales) favored by high humidity and moderate temperatures.',
        'treatment': [
            'Remove infected leaves',
            'Apply fungicide',
            'Improve air circulation',
            'Reduce humidity around plants'
        ],
        'prevention': [
            'Space plants properly',
            'Avoid overhead watering',
            'Maintain good air circulation',
            'Use resistant varieties'
        ]
    },
    2: {
        'name': 'Black Spot',
        'description': 'Fungal disease causing black spots with yellow halos on leaves.',
        'symptoms': 'Dark circular spots with fringed edges, often surrounded by yellow halos.',
        'causes': 'Fungal pathogen (Diplocarpon rosae) thriving in wet conditions.',
        'treatment': [
            'Remove infected leaves',
            'Apply appropriate fungicide',
            'Improve air circulation',
            'Modify watering practices'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Avoid wetting leaves',
            'Space plants properly',
            'Clean up fallen leaves'
        ]
    },
    3: {
        'name': 'Bacterial Leaf Spot',
        'description': 'Bacterial infection causing water-soaked spots on leaves.',
        'symptoms': 'Water-soaked spots that turn brown, often with yellow halos.',
        'causes': 'Various bacteria species, spread by water splash and high humidity.',
        'treatment': [
            'Remove infected plant parts',
            'Apply copper-based bactericide',
            'Improve air circulation',
            'Avoid overhead watering'
        ],
        'prevention': [
            'Use disease-free seeds',
            'Rotate crops',
            'Maintain proper spacing',
            'Keep leaves dry'
        ]
    },
    4: {
        'name': 'Rust',
        'description': 'Fungal disease causing rusty orange spots on leaves.',
        'symptoms': 'Orange-brown pustules on leaf undersides, yellow spots on upper surfaces.',
        'causes': 'Various rust fungi (Pucciniales) favored by wet conditions.',
        'treatment': [
            'Remove infected leaves',
            'Apply fungicide',
            'Improve air circulation',
            'Reduce humidity'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Space plants properly',
            'Avoid overhead watering',
            'Clean up plant debris'
        ]
    },
    5: {
        'name': 'Leaf Blight',
        'description': 'Fungal disease causing large brown patches on leaves.',
        'symptoms': 'Large brown patches, often starting from leaf edges or tips.',
        'causes': 'Various fungal pathogens, often spread by wind and rain.',
        'treatment': [
            'Remove infected leaves',
            'Apply appropriate fungicide',
            'Improve air circulation',
            'Reduce leaf wetness'
        ],
        'prevention': [
            'Maintain proper spacing',
            'Avoid overhead watering',
            'Use disease-resistant varieties',
            'Regular pruning'
        ]
    },
    6: {
        'name': 'Anthracnose',
        'description': 'Fungal disease causing dark, sunken lesions on leaves.',
        'symptoms': 'Dark, sunken lesions with distinct margins, often along leaf veins.',
        'causes': 'Various Colletotrichum species, favored by warm, wet conditions.',
        'treatment': [
            'Remove infected plant parts',
            'Apply fungicide',
            'Improve air circulation',
            'Reduce humidity'
        ],
        'prevention': [
            'Use disease-free seeds',
            'Maintain proper spacing',
            'Avoid overhead watering',
            'Clean up plant debris'
        ]
    },
    7: {
        'name': 'Downy Mildew',
        'description': 'Fungal-like disease causing yellow patches and white growth.',
        'symptoms': 'Yellow patches on upper leaf surface, white fuzzy growth underneath.',
        'causes': 'Oomycete pathogens, favored by cool, wet conditions.',
        'treatment': [
            'Remove infected leaves',
            'Apply appropriate fungicide',
            'Improve air circulation',
            'Reduce humidity'
        ],
        'prevention': [
            'Use resistant varieties',
            'Avoid overhead watering',
            'Maintain good air circulation',
            'Space plants properly'
        ]
    },
    8: {
        'name': 'Leaf Curl',
        'description': 'Viral disease causing leaf curling and distortion.',
        'symptoms': 'Leaves curl, twist, and become distorted, often with color changes.',
        'causes': 'Various viruses, often spread by insects.',
        'treatment': [
            'Remove infected plants',
            'Control insect vectors',
            'Use virus-free planting material',
            'Maintain plant health'
        ],
        'prevention': [
            'Use virus-free plants',
            'Control insect pests',
            'Maintain plant health',
            'Regular monitoring'
        ]
    },
    9: {
        'name': 'Leaf Scorch',
        'description': 'Physiological disorder causing leaf edges to brown and die.',
        'symptoms': 'Leaf edges turn brown and crispy, often spreading inward.',
        'causes': 'Environmental stress, water imbalance, or nutrient issues.',
        'treatment': [
            'Adjust watering schedule',
            'Improve soil drainage',
            'Apply balanced fertilizer',
            'Provide shade if needed'
        ],
        'prevention': [
            'Maintain proper watering',
            'Ensure good soil drainage',
            'Protect from extreme weather',
            'Regular soil testing'
        ]
    }
}

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

class LeafDiseaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(LeafDiseaseModel, self).__init__()
        # Use a pre-trained ResNet model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Replace the final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model():
    """Load the trained model"""
    try:
        # Create model instance
        model = LeafDiseaseModel()
        
        # Load state dict
        model_path = os.path.join('models', 'leaf_disease_model.pth')
        if not os.path.exists(model_path):
            st.error("Model file not found. Please ensure the model file exists in the models directory.")
            return None
            
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        # Convert PIL Image to tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)
        return img_tensor
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def analyze_features(image_tensor):
    """Analyze image features for disease detection"""
    try:
        # Convert tensor to numpy array
        img_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Calculate color features
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # Green health (higher is better)
        green_mask = cv2.inRange(hsv, (35, 20, 20), (85, 255, 255))
        green_health = np.mean(green_mask) / 255.0
        
        # Rust color detection
        rust_mask = cv2.inRange(hsv, (0, 50, 50), (30, 255, 255))
        rust_color = np.mean(rust_mask) / 255.0
        
        # White patches detection
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_patches = np.mean(white_mask) / 255.0
        
        # Dark spots detection
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        dark_spots = np.mean(dark_mask) / 255.0
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_intensity = np.mean(edges) / 255.0
        
        # Texture analysis
        texture_variance = np.var(gray) / 255.0
        
        # Overall health score
        health_score = (green_health * 0.6 + 
                       (1 - dark_spots) * 0.2 + 
                       (1 - white_patches) * 0.2)
        
        return {
            'green_health': green_health,
            'rust_color': rust_color,
            'white_patches': white_patches,
            'dark_spots': dark_spots,
            'edge_intensity': edge_intensity,
            'texture_variance': texture_variance,
            'health_score': health_score
        }
        
    except Exception as e:
        st.error(f"Error during feature analysis: {str(e)}")
        return {
            'green_health': 0,
            'rust_color': 0,
            'white_patches': 0,
            'dark_spots': 0,
            'edge_intensity': 0,
            'texture_variance': 0,
            'health_score': 0
        }

def get_disease_description(disease):
    """Get detailed description for each disease"""
    descriptions = {
        'Powdery Mildew': 'White powdery patches on leaves, stems, and flowers. Common in humid conditions.',
        'Rust': 'Orange or brown powdery spots on leaves. Often appears in circular patterns.',
        'Black Spot': 'Black circular spots with yellow halos. Common in roses and other plants.',
        'Bacterial Spot': 'Small, dark, water-soaked spots that may have yellow halos.',
        'Early Blight': 'Dark brown spots with concentric rings. Usually starts on lower leaves.',
        'Late Blight': 'Dark, water-soaked lesions that spread quickly in cool, wet conditions.',
        'Leaf Mold': 'Yellow patches on upper leaf surface with gray mold underneath.',
        'Septoria Leaf Spot': 'Small, circular spots with gray centers and dark borders.',
        'Healthy': 'No signs of disease. Plant shows normal, healthy growth patterns.'
    }
    return descriptions.get(disease, 'No description available.')

def predict_disease(model, image_tensor):
    """Predict disease using the model with enhanced accuracy"""
    if model is None:
        st.error("Model not loaded. Please try again.")
        return None

    try:
        # Get model predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        # Convert image for feature analysis
        img_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Analyze image features
        features = analyze_features(image_tensor)
        
        # Disease-specific detection logic with weighted scores
        disease_scores = {}
        
        # Powdery Mildew Detection (Class 1)
        if features['white_patches'] > 0.25 or features['texture_variance'] > 0.35:
            disease_scores['Powdery Mildew'] = probabilities[1].item() * 1.3
        
        # Black Spot Detection (Class 2)
        if features['dark_spots'] > 0.28 or features['edge_intensity'] > 0.55:
            disease_scores['Black Spot'] = probabilities[2].item() * 1.4
        
        # Bacterial Leaf Spot Detection (Class 3)
        if features['dark_spots'] > 0.22 and features['edge_intensity'] > 0.42:
            disease_scores['Bacterial Leaf Spot'] = probabilities[3].item() * 1.3
        
        # Rust Detection (Class 4)
        if features['rust_color'] > 0.25 or features['texture_variance'] > 0.5:
            disease_scores['Rust'] = probabilities[4].item() * 1.5
        
        # Leaf Blight Detection (Class 5)
        if features['dark_spots'] > 0.32 and features['texture_variance'] > 0.32:
            disease_scores['Leaf Blight'] = probabilities[5].item() * 1.35
        
        # Anthracnose Detection (Class 6)
        if features['dark_spots'] > 0.38 and features['edge_intensity'] > 0.48:
            disease_scores['Anthracnose'] = probabilities[6].item() * 1.4
        
        # Downy Mildew Detection (Class 7)
        if features['white_patches'] > 0.28 and features['texture_variance'] > 0.38:
            disease_scores['Downy Mildew'] = probabilities[7].item() * 1.35
        
        # Leaf Curl Detection (Class 8)
        if features['edge_intensity'] > 0.42 and features['texture_variance'] > 0.32:
            disease_scores['Leaf Curl'] = probabilities[8].item() * 1.25
        
        # Leaf Scorch Detection (Class 9)
        if features['edge_intensity'] > 0.52 and features['dark_spots'] > 0.32:
            disease_scores['Leaf Scorch'] = probabilities[9].item() * 1.25
        
        # Healthy Detection (Class 0)
        if (features['green_health'] > 0.78 and 
            features['health_score'] > 0.88 and 
            features['edge_intensity'] < 0.22):
            disease_scores['Healthy'] = probabilities[0].item() * 1.6
        
        # Ensure all diseases are considered even if specific features aren't met strongly
        # Initialize or re-initialize with raw probabilities as a baseline
        if not disease_scores or max(disease_scores.values()) < 0.4: # Lower fallback threshold
            # Populate with raw probabilities for all classes
            disease_scores = {
                CLASSES[0]['name']: probabilities[0].item(),
                CLASSES[1]['name']: probabilities[1].item(),
                CLASSES[2]['name']: probabilities[2].item(),
                CLASSES[3]['name']: probabilities[3].item(),
                CLASSES[4]['name']: probabilities[4].item(),
                CLASSES[5]['name']: probabilities[5].item(),
                CLASSES[6]['name']: probabilities[6].item(),
                CLASSES[7]['name']: probabilities[7].item(),
                CLASSES[8]['name']: probabilities[8].item(),
                CLASSES[9]['name']: probabilities[9].item()
            }
        
        # Sort diseases by score in descending order
        sorted_diseases = sorted(disease_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Get the top 3 predictions
        top_3_predictions = []
        for disease_name, score in sorted_diseases[:3]:
            disease_info = None
            # Find the correct disease information from CLASSES dictionary
            for class_id, class_data in CLASSES.items():
                if class_data['name'] == disease_name:
                    disease_info = class_data
                    break
            
            if disease_info:
                top_3_predictions.append({
                    'disease': disease_name,
                    'confidence': score,
                    'description': disease_info['description'],
                    'symptoms': disease_info['symptoms'],
                    'causes': disease_info['causes'],
                    'treatment': disease_info['treatment'],
                    'prevention': disease_info['prevention']
                })
        
        # Get the primary predicted disease (highest score)
        predicted_disease = top_3_predictions[0]['disease']
        confidence = top_3_predictions[0]['confidence']
        disease_info = top_3_predictions[0]
        
        return {
            'disease': predicted_disease,
            'confidence': confidence,
            'description': disease_info['description'],
            'symptoms': disease_info['symptoms'],
            'causes': disease_info['causes'],
            'treatment': disease_info['treatment'],
            'prevention': disease_info['prevention'],
            'all_probabilities': disease_scores,
            'top_3': top_3_predictions
        }
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="🌿",
        layout="wide"
    )
    
    # Initialize model in session state if not already done
    if st.session_state.model is None:
        st.session_state.model = load_model()
        if st.session_state.model is not None:
            st.session_state.model_loaded = True
    
    # Title and description
    st.title("🌿 leaf Disease Detection")
    st.markdown("""
    Upload an image of a plant leaf to detect diseases. The model can identify various leaf diseases 
    including Powdery Mildew, Black Spot, Bacterial Leaf Spot, and more.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image
            image_tensor = preprocess_image(image)
            
            if st.session_state.model is not None:
                    # Make prediction
                result = predict_disease(st.session_state.model, image_tensor)
                    
                if result is not None:
                    # Display results
                    st.success(f"Predicted Disease: {result['disease']}")
                    st.info(f"Confidence: {result['confidence']:.2%}")
                    
                    # Disease details in expandable sections
                    with st.expander("Description", expanded=True):
                        st.write(result['description'])
                    
                    with st.expander("Symptoms", expanded=True):
                        st.write(result['symptoms'])
                    
                    with st.expander("Causes"):
                        st.write(result['causes'])
                    
                    with st.expander("Treatment"):
                        st.write(result['treatment'])
                    
                    with st.expander("Prevention"):
                        st.write(result['prevention'])
                        
                    st.subheader("Other Potential Predictions:")
                    for i, top_pred in enumerate(result['top_3'][1:]):
                        st.write(f"{i+2}. {top_pred['disease']} (Confidence: {top_pred['confidence']:.2%})")
                        with st.expander(f"Details for {top_pred['disease']}"):
                            st.write(f"Description: {top_pred['description']}")
                            st.write(f"Symptoms: {top_pred['symptoms']}")
                            st.write(f"Causes: {top_pred['causes']}")
                            st.write(f"Treatment: {top_pred['treatment']}")
                            st.write(f"Prevention: {top_pred['prevention']}")
                else:
                    st.error("Could not make a prediction. Please try a different image.")
            else:
                st.error("Model not loaded. Please try again.")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please try a different image.")

if __name__ == "__main__":
    main() 