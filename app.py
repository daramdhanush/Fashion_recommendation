import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import logging

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        logging.error(f"Error saving file: {e}")
        return 0

def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        logging.error(f"Error extracting features: {e}")
        return None

def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        logging.error(f"Error in recommendation: {e}")
        return None

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image, width=300, caption="Your image")
        # Feature extract
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        if features is not None:
            # Recommendation
            indices = recommend(features, feature_list)
            if indices is not None:
                # Display
                st.write("Recommended images")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    try:
                        image_path = filenames[indices[0][i]]
                        absolute_image_path = os.path.abspath(image_path)
                        with col:
                            st.image(absolute_image_path, width=90)
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                        logging.error(f"Error displaying image: {e}")
                        logging.error(f"Attempted path: {absolute_image_path}")
    else:
        st.header("Some error occurred in file upload")
