# 🐾 Cat vs. Dog Image Classifier

This is a Deep Learning web application that classifies images into **Cats** or **Dogs** using a Custom Convolutional Neural Network (CNN).


## Live Demo
You can try the live version of this app on Streamlit: 
https://catsdogsclassifier.streamlit.app/

## Model Architecture

The model was built using **TensorFlow/Keras** and features:
- **Convolutional Layers (Conv2D)**: To extract spatial features from images.
- **Max Pooling Layers**: To reduce dimensionality.
- **Batch Normalization**: To speed up training and provide stability.
- **Dropout**: To prevent overfitting.
- **Dense Layers**: For final classification.
- **Activation Function**: Sigmoid (Binary Classification).

## Dataset
The model was trained on the famous **Kaggle Dogs vs. Cats dataset**, containing 25,000 images.
- **Input Size**: 150x150 pixels.
- **Channels**: 3 (RGB).
- **Labels**: Dog (0), Cat (1).

## Tech Stack
- **Python 3.10**
- **TensorFlow/Keras 3** (Deep Learning framework)
- **Streamlit** (Web Interface)
- **Pillow (PIL)** (Image Processing)
- **NumPy** (Numerical Operations)

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AhmedKamel200058/Cats_Dogs_Classifier.git
   cd Cats_Dogs_Classifier
   ```

2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
3. **Run the app:**
	```bash
	streamlit run app.py
	```


## Project Structure 📂

CatsDogsClassifier/
├── app.py                # Main script to run the Streamlit app
├── model.h5              # The pre-trained CNN model.
├── requirements.txt      # List of dependencies for the project
├── README.md             # Project documentation (this file)


