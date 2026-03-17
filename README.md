# Smart Leaf Disease Detection

## Overview
Smart Leaf Disease Detection is a Streamlit-based web application that leverages deep learning to detect plants and identify diseases from leaf images. It uses a YOLO model to detect and crop leaves from the uploaded image, and a DenseNet model for accurate disease classification. Furthermore, it incorporates Google Gemini AI to supply specialized, multilingual educational treatment recommendations.

## Features
- **Leaf Detection:** Uses a YOLO model to accurately find and box leaves in images.
- **Disease Classification:** Analyzes leaves using a custom DenseNet121 model trained on 37 different plant and disease categories.
- **Multilingual AI Advisory:** Integrates Gemini AI to provide actionable treatment advice and preventive measures in 12 languages.
- **Interactive UI:** Clean, intuitive interface built with Streamlit.

## Tech Stack
- **Frontend & UI:** Streamlit
- **Deep Learning & Vision:** PyTorch, torchvision (DenseNet121), YOLO (Ultralytics), OpenCV
- **AI Advisory:** Google Gemini AI

## Setup & Run

1. Clone the repository and navigate to the project root.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
5. Ensure the required model files (`yolo_leaf_detector.pt` and `densenet_plant_disease.pth`) are placed in the root directory. *(These files are too large for this repository and must point to valid references or be downloaded separately).*
6. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
