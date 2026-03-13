import streamlit as st
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="Smart Leaf Disease Detection", page_icon="🌿", layout="wide")

# =================================================
# LOAD ENV VARIABLES
# =================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
    st.error("Gemini API key not found. Please set it in the .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# =================================================
# CLASS NAMES (MUST MATCH TRAINING ORDER)
# =================================================
CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
    'Apple___healthy','Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___healthy'
]

# =================================================
# MODELS & UTILS
# =================================================
@st.cache_resource
def load_densenet():
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 37)
    model.load_state_dict(torch.load("densenet_plant_disease.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_yolo():
    return YOLO("yolo_leaf_detector.pt")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def parse_prediction(predicted_class):
    """Splits the raw class name into plant and disease component."""
    if '___' in predicted_class:
        plant, disease = predicted_class.split('___')
        plant = plant.replace('_', ' ').strip()
        disease = disease.replace('_', ' ').strip()
        return plant, disease
    return predicted_class.replace('_', ' ').strip(), "Unknown"

def gemini_advisory(plant_name, disease_name, language="English"):
    prompt = f"""
You are an agricultural advisory assistant.

Detected Plant: {plant_name}
Detected Disease: {disease_name}

Provide the response entirely in the {language} language. Include:
1. Short explanation of the disease
2. Common treatment or management approaches
3. Preventive measures

Rules:
- Educational advice only
- No dosage or chemical concentration
- No guarantees
- Recommend consulting agricultural experts
- Simple bullet points
- MUST be written in {language}
"""
    model = genai.GenerativeModel("gemini-3-flash-preview")
    # Using try-except to avoid app crashing if API fails
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate advisory at this time. Error: {str(e)}"

# =================================================
# UI COMPONENTS
# =================================================
def render_header():
    st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🌿 Smart Leaf Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #558b2f; margin-bottom: 30px;'>Advanced YOLO + DenseNet Analysis with Gemini AI Advisory</h4>", unsafe_allow_html=True)
    st.markdown("---")

def display_results(image, img_np, yolo_model, densenet_model, transform):
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📷 Image Analysis")
        # YOLO Detection processing
        with st.spinner("Detecting leaf..."):
            results = yolo_model(img_np)
        
        box_found = False
        crop = img_np
        if len(results[0].boxes) > 0:
            box_found = True
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box on the original image copy
            img_with_box = img_np.copy()
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                img_with_box, "Leaf Detected",
                (x1, max(10, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )
            crop = img_np[y1:y2, x1:x2]
            st.image(img_with_box, caption="YOLO Leaf Detection", use_container_width=True)
        else:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.warning("Could not detect a distinct leaf in the image. Using full image for classification.")

    with col2:
        st.subheader("📊 Diagnostic Results")
        with st.spinner("Classifying disease using DenseNet..."):
            crop_pil = Image.fromarray(crop)
            input_tensor = transform(crop_pil).unsqueeze(0)
            
            with torch.no_grad():
                outputs = densenet_model(input_tensor)
                _, pred = torch.max(outputs, 1)
            
            prediction = CLASS_NAMES[pred.item()]
            plant_name, disease_name = parse_prediction(prediction)

        # Attractive styling for results
        is_healthy = 'healthy' in disease_name.lower()
        border_color = '#22c55e' if is_healthy else '#ef4444'
        bg_color = '#f0fdf4' if is_healthy else '#fef2f2'
        text_color = '#15803d' if is_healthy else '#b91c1c'

        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {border_color}; margin-bottom: 20px;">
            <h3 style="color: #1f2937; margin-top: 0;">🌱 Plant: <b>{plant_name}</b></h3>
            <h4 style="color: {text_color}; margin-bottom: 0;">
                {'✅' if is_healthy else '🩸'} Condition: <b>{disease_name}</b>
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("💡 AI Smart Advisory")
        
        st.markdown("""
        <style>
        /* Make cursor a pointer over the selectbox */
        div[data-baseweb="select"] {
            cursor: pointer !important;
        }
        div[data-baseweb="select"] * {
            cursor: pointer !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        language = st.selectbox("Select Language for Advisory:", 
                                ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Marathi", "Bengali", "Gujarati", "Punjabi", "Odia", "Urdu"], index=0)
        
        if is_healthy:
            st.success(f"Great news! Your {plant_name} plant looks completely healthy. Keep up the good work providing it with proper care!")
        else:
            with st.spinner(f"Generating specialized treatment recommendations in {language}..."):
                advisory_text = gemini_advisory(plant_name, disease_name, language)
            
            with st.expander("Expand to read AI Recommendations", expanded=True):
                st.markdown(advisory_text)
            
            st.caption("⚠️ **DISCLAIMER:** This AI-generated advisory is for educational purposes only. It does NOT replace professional agricultural consultation. Always verify recommendations with certified agricultural experts.")

# =================================================
# MAIN APPLICATION
# =================================================
def main():
    render_header()
    
    # Load Models
    with st.spinner("Loading AI Models..."):
        densenet_model = load_densenet()
        yolo_model = load_yolo()
    
    # Center the upload component
    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown("<h3 style='text-align: center;'>📸 Upload Leaf Image</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("---")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        
        # Display Results
        display_results(image, img_np, yolo_model, densenet_model, transform)
    else:
        st.info("� Please upload an image above to begin the analysis.")

if __name__ == "__main__":
    main()
