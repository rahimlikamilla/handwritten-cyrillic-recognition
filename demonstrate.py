# Imports

import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from dataset_utils import *
from vocab_utils import *
from model_utils import CRNN


st.title("📝 Cyrillic Handwriting Recognition")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# state_dict = torch.load("crnn_cyrillic.pth", map_location=device)
state_dict = torch.load("crnn_cyrillic_50.pth", map_location=device)


# Rebuilding the model
num_classes = len(idx2char)
model = CRNN(num_classes=num_classes).to(device)

# Load weights
model.load_state_dict(state_dict)
model.eval()

uploaded_file = st.file_uploader("Upload a handwritten word image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # greyscale
    np_image = np.array(image)
    model_expected_width = 748

    # preprocess (resize + pad)
    resized = resize_with_aspect(np_image, 64)
    padded = pad_to_width(resized, model_expected_width)
    tensor = transform_pipeline(padded).unsqueeze(0).to(device)

    # Forward pass -> model prediction
    with torch.no_grad():
        output = model(tensor)
        pred_text = greedy_decode(output, idx2char)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="🖼️ Original Image", use_container_width=True)
    with col2:
        st.image(padded, caption="⚙️ Preprocessed Image (Model Input)", use_container_width=True, clamp=True)

    st.success(f"**Predicted Text:** {pred_text}")
    #
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    # st.success(f"**Predicted Text:** {pred_text}")
