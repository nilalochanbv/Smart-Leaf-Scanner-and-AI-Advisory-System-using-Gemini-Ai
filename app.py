import streamlit as st
import google.generativeai as genai
from PIL import Image

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="Smart Leaf AI", page_icon="🌿", layout="wide")

# =================================================
# LOAD API KEY (Streamlit Secrets)
# =================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# =================================================
# GEMINI VISION FUNCTION
# =================================================
def analyze_leaf(image, language="English"):
    prompt = f"""
You are a plant disease expert.

Analyze the given leaf image and provide:

1. Plant name
2. Whether it is healthy or diseased
3. If diseased, name of the disease
4. Treatment suggestions
5. Preventive measures

Rules:
- Keep answer simple
- Use bullet points
- Respond in {language}
- If unsure, say "Not clearly identifiable"
"""

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content([prompt, image])
    return response.text

# =================================================
# UI
# =================================================
def main():

    st.markdown("<h1 style='text-align: center;'>🌿 Smart Leaf AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>AI Plant Disease Detection using Gemini Vision</h4>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Upload Leaf Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    with col2:
        st.subheader("🌐 Select Language")
        language = st.selectbox(
            "",
            ["English","Tamil","Hindi","Telugu","Kannada","Malayalam","Marathi","Bengali"]
        )

    st.markdown("---")

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        with st.spinner("🔍 Analyzing using Gemini AI..."):
            result = analyze_leaf(image, language)

        st.markdown("### 🌿 AI Analysis Result")
        st.markdown(result)

        st.info("⚠️ This AI advisory is for educational purposes only. Consult agricultural experts for accurate treatment.")

    else:
        st.info("📤 Please upload a leaf image to begin analysis.")

# =================================================
# RUN
# =================================================
if __name__ == "__main__":
    main()