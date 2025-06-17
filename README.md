# Plant Disease Detector

A web application that helps identify plant diseases from leaf images using color analysis.

## Features

- Real-time disease detection
- Color-based analysis using HSV color space
- Multiple disease detection capabilities
- Confidence scores for each diagnosis
- Treatment and prevention recommendations
- User-friendly interface

## Supported Diseases

- Early Blight
- Late Blight
- Leaf Spot
- Powdery Mildew
- Healthy Plant Detection

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to http://localhost:8501

## Usage

1. Upload a clear image of a plant leaf
2. Click the "Analyze Leaf" button
3. View the results:
   - Primary diagnosis with confidence score
   - Detailed symptoms
   - Treatment recommendations
   - Prevention guidelines
   - Other possible diseases

## Best Practices

- Use well-lit images
- Focus on affected areas
- Include both healthy and diseased parts
- Ensure clear, high-resolution images
- Clean lens before taking photos

## Technical Details

The application uses:
- OpenCV for image processing
- HSV color space for better color analysis
- Streamlit for the web interface
- PIL for image handling

## Requirements

- Python 3.7+
- Streamlit
- OpenCV-Python
- NumPy
- Pillow 