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
# This class acts as a filter that strips away 'quantization_config' before Keras sees it
class FixedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None) # Remove the problematic key
        super().__init__(*args, **kwargs)

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    try:
        # We tell Keras: Whenever you see 'Dense' in the file, use our 'FixedDense' instead
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

# --- UI Layout ---
st.title("🐾 Cat vs. Dog Classifier")
st.markdown("Upload an image, and the AI will tell you if it's a Cat or a Dog!")

uploaded_file = st.file_uploader("Select an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width='stretch')
    
    if st.button('Predict'):
        if model is not None:
            with st.spinner('Analyzing...'):
                # Preprocessing
                img_rgb = image.convert('RGB')
                img_resized = img_rgb.resize((150, 150))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                prediction = model.predict(img_array, verbose=0)[0]

                # Probabilities (Cat=1, Dog=0)
                prob_cat = float(prediction[1])
                prob_dog = 1 - prob_cat

                # Results
                st.subheader("Results:")
                st.write(f"Cat 🐱: {prob_cat*100:.2f}%")
                st.progress(prob_cat)
                st.write(f"Dog 🐶: {prob_dog*100:.2f}%")
                st.progress(prob_dog)

                if prob_cat > 0.5:
                    st.success(f"Final Verdict: It's a CAT!")
                else:
                    st.success(f"Final Verdict: It's a DOG!")
        else:
            st.error("Model failed to load. Please check the technical logs.")

st.markdown("---")
st.caption("Developed for DEBI AI Track - 2026")
