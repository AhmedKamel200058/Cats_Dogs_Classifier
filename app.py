import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(page_title="🐾 Cat vs. Dog Classifier", layout="centered")

# --- Environment Setup ---
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('model.keras', compile=False, safe_mode=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

st.title("🐾 Cat vs. Dog Classifier")

# --- Image Upload ---
uploaded_file = st.file_uploader("Select an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    
    st.image(image, caption='Uploaded Image', width='stretch')
    
    if st.button('Predict'):
        if model is not None:
            with st.spinner('Processing...'):
                
                img_rgb = image.convert('RGB')
                
                # Preprocessing
                IMAGE_SIZE = (150, 150)
                img_resized = img_rgb.resize(IMAGE_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0) # Result: (1, 150, 150, 3)

                # Prediction
                prediction = model.predict(img_array, verbose=0)[0]

                # Cat=1, Dog=0 logic
                prob_cat = float(prediction[1])
                prob_dog = 1 - prob_cat

                st.subheader("Results:")
                st.write(f"Cat 🐱: {prob_cat*100:.2f}%")
                st.progress(prob_cat)
                st.write(f"Dog 🐶: {prob_dog*100:.2f}%")
                st.progress(prob_dog)

                if prob_cat > 0.5:
                    st.success(f"Final Perdict: It's a CAT!")
                else:
                    st.success(f"Final Perdict: It's a DOG!")
