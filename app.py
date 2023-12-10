import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/yashp/Downloads/my_model2.h5') # Replace with your model's path

# Define class labels
class_labels = ['Non-Vehicle', 'Vehicle']

# Create a Streamlit web app
st.title('Vehicle Detection App')

st.write('Upload an image to classify whether it contains a vehicle or not.')

# Upload an image for classification
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    image = Image.open(uploaded_image)
    image = image.resize((64, 64))  # Adjust the size as per your model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize the image data

    # Classify the image
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the result
    st.write(f'Prediction: {predicted_class}')
