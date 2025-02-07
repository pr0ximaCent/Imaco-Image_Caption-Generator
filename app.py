import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import pickle
from PIL import Image
import io
import time

# Configure Tensorflow to use GPU memory growth to avoid memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Cache the models and tokenizer in session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

def load_models():
    if not st.session_state.models_loaded:
        with st.spinner('Loading models... (this will only happen once)'):
            # Load caption model
            st.session_state.caption_model = load_model('caption_model.h5', compile=False)
            
            # Load and configure VGG16
            base_model = VGG16()
            st.session_state.vgg_model = tf.keras.Model(inputs=base_model.inputs, 
                                                       outputs=base_model.layers[-2].output)
            
            # Load tokenizer
            with open('tokenizer.pkl', 'rb') as f:
                st.session_state.tokenizer = pickle.load(f)
            
            st.session_state.models_loaded = True
            
            # Warmup the models with a dummy prediction
            dummy_image = np.zeros((1, 224, 224, 3))
            _ = st.session_state.vgg_model.predict(dummy_image, verbose=0)
            
@st.cache_data
def extract_features(image_array):
    """Cache feature extraction for identical images"""
    features = st.session_state.vgg_model.predict(image_array, verbose=0)
    return features

def process_image(image):
    # Resize image to 224x224
    image = image.resize((224, 224))
    # Convert PIL image to numpy array
    image = img_to_array(image)
    # Reshape for VGG
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preprocess image for VGG
    image = preprocess_input(image)
    return image

def generate_caption(image, max_length=34):
    # Process image
    processed_image = process_image(image)
    
    # Extract features with progress bar
    with st.spinner('Extracting image features...'):
        features = extract_features(processed_image)
    
    # Initialize caption generation
    in_text = 'startseq'
    
    with st.spinner('Generating caption...'):
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Generate caption word by word
        for i in range(max_length):
            # Update progress
            progress = (i + 1) / max_length
            progress_bar.progress(progress)
            
            # Encode the current input text
            sequence = st.session_state.tokenizer.texts_to_sequences([in_text])[0]
            # Pad the sequence
            sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
            
            # Predict next word
            yhat = st.session_state.caption_model.predict([features, sequence], verbose=0)
            # Get index with highest probability
            yhat = np.argmax(yhat)
            
            # Convert index to word
            word = None
            for word, index in st.session_state.tokenizer.word_index.items():
                if index == yhat:
                    break
            
            # Stop if word not found
            if word is None:
                break
                
            # Append word to caption
            in_text += ' ' + word
            
            # Stop if end sequence found
            if word == 'endseq':
                break
    
    # Remove progress bar after completion
    progress_bar.empty()
    
    # Remove start and end tokens
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

def main():
    st.title('Image Caption Generator')
    st.write('Upload an image and get its caption!')
    
    # Load models at startup
    try:
        load_models()
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Generate caption button
            if st.button('Generate Caption'):
                try:
                    start_time = time.time()
                    
                    # Generate caption
                    caption = generate_caption(image)
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Display results
                    st.success(f'Caption generated in {processing_time:.2f} seconds!')
                    st.write('**Generated Caption:**')
                    st.write(caption)
                    
                except Exception as e:
                    st.error(f"Error generating caption: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.write("Please make sure all required model files are in the correct location.")

if __name__ == '__main__':
    main()
