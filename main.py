import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import tempfile
import os
import tensorflow as tf

# Function to generate and return caption
def generate_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, img_size=224):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    image_features = feature_extractor.predict(img_array, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index)
        if word is None or word == "endseq":
            break
        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()

# Streamlit app interface
def main():
    st.title("🖼️ Image Caption Generator")
    st.write("Upload an image and generate a caption using a CNN-LSTM model.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_image.read())
            tmp_path = tmp_file.name

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        model_path = "/Users/pranavbejgam/Desktop/Image-Caption-Generator-Using-Deep-Learning-Image-Captioning-Using-CNN-LSTM-main/models/model.keras"
        tokenizer_path = "/Users/pranavbejgam/Desktop/Image-Caption-Generator-Using-Deep-Learning-Image-Captioning-Using-CNN-LSTM-main/models/tokenizer.pkl"
        feature_extractor_path = "/Users/pranavbejgam/Desktop/Image-Caption-Generator-Using-Deep-Learning-Image-Captioning-Using-CNN-LSTM-main/models/feature_extractor.keras"

        with st.spinner("Generating caption..."):
            caption = generate_caption(tmp_path, model_path, tokenizer_path, feature_extractor_path)
            st.success("Caption Generated!")
            st.markdown(f"**Caption**: {caption}")

        os.remove(tmp_path)  # Clean up temp file

        


if __name__ == "__main__":
    main()
