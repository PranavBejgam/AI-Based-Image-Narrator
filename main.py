import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import tempfile
import os
import gdown

def download_models():
    os.makedirs("models", exist_ok=True)

    files = {
        "models/model.h5": "https://drive.google.com/uc?id=1KepFg1i5ehBQOpdhrzmnCEqOYvxE5C1t",
        "models/feature_extractor.h5": "https://drive.google.com/uc?id=1mvEhxShS4Tyla-kBkDLpl3VR92VsWgvL",
        "models/tokenizer.pkl": "https://drive.google.com/uc?id=1STdIUk7Vh4FQQBCHmyenLdzr7VVRpQio",
    }

    for path, url in files.items():
        if not os.path.exists(path):
            with st.spinner(f"Downloading {os.path.basename(path)}..."):
                gdown.download(url, path, quiet=False)
                st.success(f"{os.path.basename(path)} downloaded!")

@st.cache_resource
def load_caption_model():
    return load_model("models/model.h5")

@st.cache_resource
def load_feature_extractor():
    return load_model("models/feature_extractor.h5")

@st.cache_resource
def load_tokenizer():
    with open("models/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

def generate_caption(image_path, max_length=34, img_size=224):
    # Load models and tokenizer only once (cached)
    caption_model = load_caption_model()
    feature_extractor = load_feature_extractor()
    tokenizer = load_tokenizer()

    # Preprocess image
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features
    image_features = feature_extractor.predict(img_array, verbose=0)

    # Generate caption
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

    caption = in_text.replace("startseq ", "").replace(" endseq", "").strip()
    return caption

def main():
    st.title("🖼️ Image Caption Generator")
    st.write("Upload an image to generate a caption using CNN + LSTM")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_image.read())
            tmp_path = tmp_file.name

        # Show the image (using use_column_width instead of deprecated use_container_width)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Download models if needed
        download_models()

        with st.spinner("Generating caption..."):
            try:
                caption = generate_caption(tmp_path)
                st.success("Caption Generated!")
                st.markdown(f"**Caption**: {caption}")
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")

        # Clean up
        os.remove(tmp_path)

if __name__ == "__main__":
    main()