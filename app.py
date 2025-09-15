import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os # Import the os module

# --- Functions ---

# Function to load and inject CSS robustly
def load_css(file_name):
    # Check if the file exists before trying to open it
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # You can optionally print a warning to the console if the file is not found
        print(f"Warning: CSS file '{file_name}' not found. Using default styles.")

# Cache the model to load it only once
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

# Function to preprocess and predict
def predict_image(model, image):
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(processed_img)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# --- Main App Logic ---

# Load CSS and Model
load_css("style.css")
model = load_model()

# --- Streamlit App UI ---

# Display title and instructions
st.title("🖼️ Universal Image Classifier")
st.write("Upload an image to see the prediction!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Logic to run after a file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    with st.spinner('Classifying...'):
        predictions = predict_image(model, image)
    
    st.subheader("Prediction Results:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        formatted_label = label.replace('_', ' ').title()
        st.write(f"{i+1}. **{formatted_label}** - Confidence: {score:.2%}")