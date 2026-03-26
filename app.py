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

# --- FIX: Custom Class to handle the quantization_config error ---
class FixedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        # Remove the problematic keyword if it exists
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# --- Model Loading with Custom Objects ---
@st.cache_resource
def load_my_model():
    try:
        # We tell Keras to use our 'FixedDense' whenever it sees 'Dense' in the file
        custom_objects = {'Dense': FixedDense}
        model = tf.keras.models.load_model(
            'model.keras', 
            custom_objects=custom_objects, 
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# --- UI and Logic ---
st.title("🐾 Cat vs. Dog Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width='stretch')
    
    if st.button('Predict'):
        if model is not None:
            with st.spinner('Analyzing...'):
                img_rgb = image.convert('RGB')
                img_resized = img_rgb.resize((150, 150))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array, verbose=0)[0]
                
                # Based on your labels: Cat=1, Dog=0
                prob_cat = float(prediction[1])
                prob_dog = 1 - prob_cat

                st.subheader("Results:")
                st.write(f"Cat 🐱: {prob_cat*100:.2f}%")
                st.progress(prob_cat)
                st.write(f"Dog 🐶: {prob_dog*100:.2f}%")
                st.progress(prob_dog)

                if prob_cat > 0.5:
                    st.success("Result: It's a CAT!")
                else:
                    st.success("Result: It's a DOG!")
