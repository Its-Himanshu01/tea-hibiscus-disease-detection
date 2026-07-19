
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load your existing trained model
model = tf.keras.models.load_model("tea_hibiscus_model.h5")

# The original 13 trained categories matching your folder dataset exactly
class_names = [
    "Hibiscus Citruspot",
    "Hibiscus Early_Mild_Spotting",
    "Hibiscus Fungal",
    "Hibiscus Healthy",
    "Hibiscus Mild_Edge_Damage",
    "Hibiscus Senescent",
    "Hibiscus Slightly_Diseased",
    "Hibiscus Wrinkled_Leaf",
    "Tea Algal Spotout",
    "Tea Brown Blight",
    "Tea Grey Blight",
    "Tea Healthy",
    "Tea Red Spot"
]

# Plant disease database lookup dictionary
disease_info = { 
    "Hibiscus Citruspot": {
        "symptoms": "Yellow to brown circular spots appear on leaves.",
        "cause": "Fungal or bacterial infection.",
        "impact": "Reduces plant health and leaf quality.",
        "treatment": "Remove infected leaves and apply suitable fungicide."
    },
    "Hibiscus Early_Mild_Spotting": {
        "symptoms": "Small light-colored spots on leaves.",
        "cause": "Early-stage disease infection.",
        "impact": "Can spread if untreated.",
        "treatment": "Monitor plants and use preventive fungicide sprays."
    },
    "Hibiscus Fungal": {
        "symptoms": "Dark patches and irregular lesions on leaves.",
        "cause": "Fungal pathogen attack.",
        "impact": "Weakens plant growth.",
        "treatment": "Apply fungicide and improve air circulation."
    },
    "Hibiscus Healthy": {
        "symptoms": "No disease symptoms detected.",
        "cause": "Healthy plant condition.",
        "impact": "Normal growth and flowering.",
        "treatment": "Maintain proper watering and nutrition."
    },
    "Hibiscus Mild_Edge_Damage": {
        "symptoms": "Leaf edges appear damaged or dry.",
        "cause": "Environmental stress or minor infection.",
        "impact": "Reduced leaf quality.",
        "treatment": "Ensure adequate watering and monitor for disease."
    },
    "Hibiscus Senescent": {
        "symptoms": "Leaves appear aged and yellowing.",
        "cause": "Natural aging process.",
        "impact": "Reduced photosynthetic activity.",
        "treatment": "Prune old leaves and maintain plant health."
    },
    "Hibiscus Slightly_Diseased": {
        "symptoms": "Minor discoloration and leaf damage.",
        "cause": "Early disease development.",
        "impact": "May progress if untreated.",
        "treatment": "Monitor closely and apply preventive treatment."
    },
    "Hibiscus Wrinkled_Leaf": {
        "symptoms": "Leaves appear curled or wrinkled.",
        "cause": "Disease, pest attack, or nutrient deficiency.",
        "impact": "Affects plant growth.",
        "treatment": "Inspect for pests and provide balanced nutrients."
    },
    "Tea Algal Spotout": {
        "symptoms": "Orange or reddish spots on leaf surface.",
        "cause": "Algal infection.",
        "impact": "Reduces photosynthesis.",
        "treatment": "Improve drainage and apply copper-based fungicide."
    },
    "Tea Brown Blight": {
        "symptoms": "Brown circular spots that enlarge over time.",
        "cause": "Fungal infection.",
        "impact": "Reduces tea leaf quality and yield.",
        "treatment": "Remove infected leaves and apply recommended fungicide."
    },
    "Tea Grey Blight": {
        "symptoms": "Grey lesions with dark margins on leaves.",
        "cause": "Fungal pathogen.",
        "impact": "Premature leaf fall and reduced growth.",
        "treatment": "Maintain field hygiene and use fungicides."
    },
    "Tea Healthy": {
        "symptoms": "No disease symptoms detected.",
        "cause": "Healthy plant.",
        "impact": "Normal growth and productivity.",
        "treatment": "Regular monitoring and proper care."
    },
    "Tea Red Spot": {
        "symptoms": "Red-colored spots scattered across leaves.",
        "cause": "Fungal disease.",
        "impact": "Can reduce leaf quality.",
        "treatment": "Remove affected leaves and apply fungicide."
    }
}

st.title("Tea & Hibiscus Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a Tea or Hibiscus leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to numpy array without arbitrary division
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # FIX: Run the MobileNetV2 preprocessing function matching Train.py perfectly
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Perform prediction
    prediction = model.predict(img_array, verbose=0)
    
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    # GATEKEEPER FILTER: Block non-leaf data (dogs, bats, etc.) using confidence boundaries
    if confidence < 80.0:
        st.error("❌ Invalid Image Input")
        st.warning("The uploaded image does not match features of a valid Tea or Hibiscus plant. Please try again with a clear leaf photo.")
        st.info(f"System Identification Certainty: {confidence:.2f}% (Required baseline: >80.0%)")
        st.stop()

    predicted_class = class_names[predicted_index]

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence Score: {confidence:.2f}%")

    if predicted_class in disease_info:
        info = disease_info[predicted_class]
        st.subheader("Disease Details")
        st.write("**Symptoms:**", info["symptoms"])
        st.write("**Cause:**", info["cause"])
        st.write("**Impact:**", info["impact"])
        st.write("**Treatment:**", info["treatment"])
