import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown
import os

# Initialize session state variables
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'caption_model' not in st.session_state:
    st.session_state.caption_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'vgg_model' not in st.session_state:
    st.session_state.vgg_model = None

# Function to get word from index
def get_word_from_index(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

# Function to download the model from Google Drive
def download_model_from_drive(file_id, output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)

# Function to load models
@st.cache_resource
def load_models():
    if not st.session_state.models_loaded:
        with st.spinner('Loading models... (this will only happen once)'):
            try:
                # Check if model is already downloaded
                if not os.path.exists('caption_model.h5'):
                    download_model_from_drive('1BuncgbkV33pGpciip3ljIc9f8nmClaEH', 'caption_model.h5')
                    st.success("Model downloaded successfully!")
                else:
                    st.success("Model is already available.")

                # Load the trained model
                model = tf.keras.models.load_model('caption_model.h5')

                # Load the tokenizer
                with open('tokenizer.pkl', 'rb') as tokenizer_file:
                    st.session_state.tokenizer = pickle.load(tokenizer_file)

                st.session_state.caption_model = model
                st.session_state.models_loaded = True

                # Load VGG16 model for feature extraction
                base_model = VGG16(weights='imagenet')
                st.session_state.vgg_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

                # Warm up the models with a dummy prediction
                dummy_image = np.zeros((1, 224, 224, 3))
                _ = st.session_state.vgg_model.predict(dummy_image, verbose=0)

            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                raise e

# Function to predict caption
def predict_caption(model, image_features, tokenizer, max_caption_length):
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        if predicted_word is None or predicted_word == "endseq":
            break
        caption += " " + predicted_word
    return caption

# Load models at startup
load_models()

# Streamlit app UI
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    try:
        st.subheader("Uploaded Image")
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        st.subheader("Generated Caption")
        with st.spinner("Generating caption..."):
            # Load and preprocess image
            image = load_img(uploaded_image, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)

            # Extract features using VGG16
            image_features = st.session_state.vgg_model.predict(image, verbose=0)

            # Generate caption
            max_caption_length = 34  # Make sure this matches your training configuration
            generated_caption = predict_caption(
                st.session_state.caption_model, 
                image_features, 
                st.session_state.tokenizer, 
                max_caption_length
            )

            # Clean up and display caption
            generated_caption = generated_caption.replace("startseq", "").replace("endseq", "").strip()

            # Display the generated caption with custom styling
            st.markdown(
                f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
                f'<p style="font-style: italic;">"{generated_caption}"</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error("Please try uploading a different image or check if the models are loaded correctly.")
