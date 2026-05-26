import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- 1. SETUP THE WEB PAGE ---
st.set_page_config(page_title="Swasth Sathi - MRI Brain Tumor Detector", layout="centered")

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI scan to detect and classify brain tumors.")

# --- 2. SIDEBAR FOR EXPLAINABILITY (For your Viva/Examiners) ---
with st.sidebar:
    st.header("About this Model")
    st.write("This application uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras.")
    st.write("**Classes Detected:**")
    st.write("- Glioma")
    st.write("- Meningioma")
    st.write("- Pituitary Tumor")
    st.write("- No Tumor (Healthy)")

# --- 3. CACHE THE MODEL ---
# This tells Streamlit to load the 18MB model only once, not every time we click a button!
@st.cache_resource
def load_disease_model():
    # Make sure this matches the exact name of the file saved in your Jupyter Notebook
    return tf.keras.models.load_model('tumor_classifier_model.h5')

# --- 4. FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an MRI image (JPG/PNG)", type=["jpg", "png", "jpeg"])

# --- 5. PREDICTION LOGIC ---
if uploaded_file is not None:
    # A. Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    
    # Add a button so the prediction doesn't happen until the user is ready
    if st.button("Analyze MRI Scan"):
        with st.spinner("Model is analyzing the scan..."):
            
            try:
                # Load our trained model
                model = load_disease_model()
                
                # B. Preprocess the Image (Must match our training pipeline exactly!)
                # 1. Resize to 150x150
                img_resized = image.resize((150, 150))
                # 2. Convert image to numbers (array)
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                # 3. Add a "batch" dimension (because our model expects batches, even if it's just 1 image)
                img_array = np.expand_dims(img_array, axis=0)
                # 4. Shrink the pixel values (0-1) just like we did in training!
                img_array = img_array / 255.0
                
                # C. Make the Prediction
                predictions = model.predict(img_array)
                
                # D. Interpret the Results
                # Our folders in 'data/Training' were alphabetized by default:
                # 0: glioma, 1: meningioma, 2: no_tumor, 3: pituitary
                class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
                
                # Find the highest probability
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_index]
                confidence = np.max(predictions[0]) * 100
                
                # E. Display the Output to the User
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"**Confidence Score:** {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"An exact error occurred: {e}")