import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Smart Leaf AI",
    page_icon="🌿",
    layout="wide"
)

# =================================================
# CUSTOM UI (Premium Look)
# =================================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #4ade80, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-text {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}

.block {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD API KEY
# =================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ API Key missing! Add in Streamlit Secrets")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# =================================================
# GEMINI FUNCTION
# =================================================
def analyze_leaf(image, language="English"):

    prompt = f"""
You are a plant disease expert.

Analyze the given leaf image and provide:

1. Plant name
2. Health status (Healthy / Diseased)
3. Disease name (if any)
4. Treatment suggestions
5. Preventive measures

Rules:
- Simple bullet points
- Easy language
- Respond in {language}
"""

    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/png",
                "data": img_byte_arr
            }
        ]
    )

    return response.text

# =================================================
# UI HEADER
# =================================================
st.markdown('<p class="main-title">🌿 Smart Leaf AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-powered Plant Disease Detection using Gemini Vision</p>', unsafe_allow_html=True)

st.divider()

# =================================================
# INPUT SECTION
# =================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📸 Upload Leaf Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown("### 🌐 Select Language")
    language = st.selectbox(
        "",
        ["English","Tamil","Hindi","Telugu","Kannada","Malayalam","Marathi","Bengali"]
    )

st.divider()

# =================================================
# OUTPUT SECTION
# =================================================
if uploaded_file:

    image = Image.open(uploaded_file)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 🔍 Uploaded Image")
        st.image(image, use_container_width=True)

    with colB:
        st.markdown("### 🤖 AI Analysis")

        with st.spinner("Analyzing with Gemini AI..."):
            result = analyze_leaf(image, language)

        st.markdown("""
        <div class="block">
        """, unsafe_allow_html=True)

        st.markdown(result)

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.info("⚠️ This advisory is AI-generated. Consult agricultural experts for accurate treatment.")

else:
    st.info("📤 Upload a leaf image to start analysis.")