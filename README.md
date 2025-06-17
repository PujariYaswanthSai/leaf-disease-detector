# Plant Leaf Disease Detector

A deep learning-based web application that helps identify plant diseases from leaf images using computer vision and machine learning techniques.

## 🌟 Features

- Real-time disease detection using deep learning
- Support for multiple plant diseases
- Detailed disease analysis and confidence scores
- Treatment and prevention recommendations
- User-friendly web interface
- Feedback system for continuous improvement
- Color-based analysis using HSV color space
- Multiple disease detection capabilities

## 🏗️ Project Structure

```
leaf-disease-detector/
├── app.py                 # Main Streamlit application
├── train_leaf_disease.py  # Training script for the model
├── download_dataset.py    # Script to download the dataset
├── setup_kaggle.py       # Kaggle API setup script
├── requirements.txt      # Python dependencies
├── templates/           # HTML templates
├── models/             # Trained model files
├── data/              # Dataset directory
├── uploads/          # Temporary upload directory
└── feedback_images/  # User feedback images
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PujariYaswanthSai/leaf-disease-detector.git
cd leaf-disease-detector
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to http://localhost:8501

## 💻 Usage

1. Upload a clear image of a plant leaf through the web interface
2. Click the "Analyze Leaf" button
3. View the results:
   - Primary diagnosis with confidence score
   - Detailed symptoms
   - Treatment recommendations
   - Prevention guidelines
   - Other possible diseases

## 🎯 Supported Diseases

The application can detect the following conditions:
- Healthy Plants
- Powdery Mildew
- Black Spot
- Bacterial Leaf Spot
- Rust
- Leaf Blight
- Anthracnose
- Downy Mildew
- Leaf Curl
- Leaf Scorch

## 🛠️ Technical Details

### Backend
- Built with Python and Streamlit
- Uses PyTorch for deep learning
- ResNet50 architecture for image classification
- OpenCV for image processing
- HSV color space for color analysis

### Frontend
- Streamlit web interface
- Real-time image processing
- Interactive results display
- User feedback system

### Dependencies
```
streamlit==1.31.1
torch==2.7.1
torchvision==0.22.1
opencv-python==4.7.0.72
pillow==9.5.0
numpy==1.24.3
requests==2.28.2
scikit-image==0.22.0
kaggle==1.5.16
```

## 📸 Best Practices for Image Capture

- Use well-lit images
- Focus on affected areas
- Include both healthy and diseased parts
- Ensure clear, high-resolution images
- Clean lens before taking photos
- Capture the entire leaf when possible
- Avoid shadows and glare

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Pujari Yaswanth Sai

## 🙏 Acknowledgments

- Kaggle for the dataset
- Streamlit for the web framework
- PyTorch team for the deep learning framework 