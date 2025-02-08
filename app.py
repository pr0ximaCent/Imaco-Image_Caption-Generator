import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Embedding, Dropout, 
                                   concatenate, Bidirectional, Dot, Activation, 
                                   RepeatVector, Multiply, Lambda)
import numpy as np
import pickle
from PIL import Image
import time

# Configure Tensorflow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def create_caption_model(vocab_size, max_length):
    """Recreate the caption model architecture"""
    
    # Features from the CNN
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder
    decoder1 = concatenate([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Create the model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# Cache models and tokenizer in session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

def load_models():
    if not st.session_state.models_loaded:
        with st.spinner('Loading models... (this will only happen once)'):
            try:
                # Load tokenizer first to get vocab_size
                with open('tokenizer.pkl', 'rb') as f:
                    st.session_state.tokenizer = pickle.load(f)
                
                # Get vocabulary size
                vocab_size = len(st.session_state.tokenizer.word_index) + 1
                max_length = 34  # Set this to the same value used during training
                
                # Create model with correct architecture
                st.session_state.caption_model = create_caption_model(vocab_size, max_length)
                
                # Load weights
                st.session_state.caption_model.load_weights('caption_model.h5')
                
                # Load and configure VGG16
                base_model = VGG16()
                st.session_state.vgg_model = Model(inputs=base_model.inputs, 
                                                 outputs=base_model.layers[-2].output)
                
                st.session_state.models_loaded = True
                
                # Warmup models
                dummy_image = np.zeros((1, 224, 224, 3))
                _ = st.session_state.vgg_model.predict(dummy_image, verbose=0)
                
            except Exception as e:
                raise Exception(f"Error loading models: {str(e)}")

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

def extract_features(image_array):
    features = st.session_state.vgg_model.predict(image_array, verbose=0)
    return features

def generate_caption(image, max_length=34):
    # Process image
    processed_image = process_image(image)
    
    # Extract features
    with st.spinner('Extracting image features...'):
        features = extract_features(processed_image)
    
    # Initialize caption generation
    in_text = 'startseq'
    
    with st.spinner('Generating caption...'):
        progress_bar = st.progress(0)
        
        # Generate caption word by word
        for i in range(max_length):
            # Update progress
            progress = (i + 1) / max_length
            progress_bar.progress(progress)
            
            # Encode current input text
            sequence = st.session_state.tokenizer.texts_to_sequences([in_text])[0]
            # Pad sequence
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
            
            if word is None:
                break
                
            in_text += ' ' + word
            
            if word == 'endseq':
                break
    
    progress_bar.empty()
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

def main():
    st.title('Image Caption Generator')
    st.write('Upload an image and get its caption!')
    
    try:
        load_models()
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Generate Caption'):
                try:
                    start_time = time.time()
                    caption = generate_caption(image)
                    processing_time = time.time() - start_time
                    
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
