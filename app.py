import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# =========================
# CONFIG
# =========================
MODEL_PATH = "model_v2_with_non_leaf.keras"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.65  # avoid wrong disease prediction

# =========================
# SAFETY CHECK
# =========================
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please check deployment.")
    st.stop()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# CLASS NAMES (ORDER MUST MATCH TRAINING)
# =========================
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "NON_LEAF",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# =========================
# MULTI-LANGUAGE UI
# =========================
LANGUAGES = {
    "English": {
        "title": "🌿 Plant Leaf Disease Detection",
        "upload": "Upload or Paste a Leaf Image",
        "non_leaf": "❌ This image is NOT a plant leaf.",
        "healthy": "✅ Leaf is HEALTHY",
        "disease": "🦠 Disease Detected",
        "confidence": "Confidence",
    },
    "தமிழ்": {
        "title": "🌿 இலை நோய் கண்டறிதல்",
        "upload": "இலை படத்தை பதிவேற்றவும் / ஒட்டவும்",
        "non_leaf": "❌ இது தாவர இலை அல்ல",
        "healthy": "✅ இலை ஆரோக்கியமாக உள்ளது",
        "disease": "🦠 நோய் கண்டறியப்பட்டது",
        "confidence": "நம்பகத்தன்மை",
    },
    "हिन्दी": {
        "title": "🌿 पत्तियों की बीमारी पहचान",
        "upload": "पत्ती की छवि अपलोड / पेस्ट करें",
        "non_leaf": "❌ यह पत्ती नहीं है",
        "healthy": "✅ पत्ती स्वस्थ है",
        "disease": "🦠 बीमारी पाई गई",
        "confidence": "विश्वास स्तर",
    },
    "తెలుగు": {
        "title": "🌿 ఆకుల వ్యాధి గుర్తింపు",
        "upload": "ఆకు చిత్రాన్ని అప్‌లోడ్ / పేస్ట్ చేయండి",
        "non_leaf": "❌ ఇది ఆకు కాదు",
        "healthy": "✅ ఆకు ఆరోగ్యంగా ఉంది",
        "disease": "🦠 వ్యాధి గుర్తించబడింది",
        "confidence": "నమ్మకం",
    },
}

language = st.sidebar.selectbox("🌐 Language / மொழி / भाषा / భాష", list(LANGUAGES.keys()))
T = LANGUAGES[language]

# =========================
# UI
# =========================
st.title(T["title"])

uploaded_file = st.file_uploader(
    T["upload"],
    type=["jpg", "jpeg", "png"]
)

# =========================
# IMAGE PROCESSING
# =========================
def preprocess(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# PREDICTION
# =========================
if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    img_array = preprocess(image)
    preds = model.predict(img_array)[0]

    class_index = np.argmax(preds)
    confidence = preds[class_index]

    predicted_class = CLASS_NAMES[class_index]

    # NON-LEAF HANDLING
    if predicted_class == "NON_LEAF":
        st.error(T["non_leaf"])
        st.info(f"{T['confidence']}: {confidence*100:.2f}%")
        st.stop()

    # CONFIDENCE CHECK
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence prediction. Image may be unclear.")
        st.info(f"{T['confidence']}: {confidence*100:.2f}%")
        st.stop()

    # SPLIT CROP & STATUS
    crop, status = predicted_class.split("___")

    st.success(f"🌱 Crop: **{crop.replace('_', ' ')}**")

    if status.lower() == "healthy":
        st.success(T["healthy"])
    else:
        st.error(f"{T['disease']}: **{status.replace('_', ' ')}**")

    st.info(f"{T['confidence']}: **{confidence*100:.2f}%**")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("AI-based Plant Disease Detection • Academic Project")
