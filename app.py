import sys
# Monkey-patch to satisfy imports for keras.src.legacy
try:
    # Import the legacy optimizers module from TensorFlow's Keras
    import tensorflow.keras.optimizers.legacy as legacy_optimizers
    # Map the missing module name to the legacy optimizers module
    sys.modules["keras.src.legacy"] = legacy_optimizers
except ImportError:
    pass

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Embedding, Dropout, 
                                   concatenate)
import numpy as np
import pickle
from PIL import Image
import time

# Rest of your code remains the same until the load_models function

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
                model = create_caption_model(vocab_size, max_length)
                
                # Use the standard Adam optimizer instead of legacy
                optimizer = keras.optimizers.Adam()
                model.compile(loss='categorical_crossentropy', optimizer=optimizer)
                
                # Load weights
                model.load_weights('caption_model.h5')
                st.session_state.caption_model = model
                
                # Load and configure VGG16
                base_model = VGG16()
                st.session_state.vgg_model = Model(inputs=base_model.inputs, 
                                                 outputs=base_model.layers[-2].output)
                
                st.session_state.models_loaded = True
                
                # Warmup the models with a dummy prediction
                dummy_image = np.zeros((1, 224, 224, 3))
                _ = st.session_state.vgg_model.predict(dummy_image, verbose=0)
                
            except Exception as e:
                raise Exception(f"Error loading models: {str(e)}")

# Rest of your code remains the same
