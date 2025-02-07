import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import pickle
from PIL import Image
import io

# Load the pre-trained model and required files
@st.cache_resource
def load_caption_model():
    return load_model('caption_model.h5')

@st.cache_resource
def load_vgg_model():
    model = VGG16()
    # Remove the last layer (output layer)
    return tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

def extract_features(image, vgg_model):
    # Resize image to 224x224
    image = image.resize((224, 224))
    # Convert PIL image to numpy array
    image = img_to_array(image)
    # Reshape for VGG
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preprocess image for VGG
    image = preprocess_input(image)
    # Extract features
    features = vgg_model.predict(image, verbose=0)
    return features

def generate_caption(image, caption_model, vgg_model, tokenizer, max_length=34):
    # Extract features from the image
    features = extract_features(image, vgg_model)
    
    # Initialize caption generation
    in_text = 'startseq'
    
    # Generate caption word by word
    for _ in range(max_length):
        # Encode the current input text
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word
        yhat = caption_model.predict([features, sequence], verbose=0)
        # Get index with highest probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = None
        for word, index in tokenizer.word_index.items():
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
    
    # Remove start and end tokens
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# Streamlit UI
def main():
    st.title('Image Caption Generator')
    st.write('Upload an image and get its caption!')
    
    # Load models
    try:
        caption_model = load_caption_model()
        vgg_model = load_vgg_model()
        tokenizer = load_tokenizer()
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Generate caption button
            if st.button('Generate Caption'):
                with st.spinner('Generating caption...'):
                    # Generate caption
                    caption = generate_caption(image, caption_model, vgg_model, tokenizer)
                    
                    # Display caption
                    st.success('Caption generated successfully!')
                    st.write('**Generated Caption:**')
                    st.write(caption)
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.write("Please make sure all required model files are in the correct location.")

if __name__ == '__main__':
    main()