import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Compatibility wrapper: handles models saved with quantization_config in Dense config
class CompatDense(keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

# Load model with compatibility fix
model = keras.models.load_model(
    "best_brain_tumor_model.h5",
    custom_objects={"Dense": CompatDense}
)

st.title("Brain Tumor Detection System")

st.write("Upload an MRI image for prediction")

uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    img = image.convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")

    st.write("Prediction Score:", float(prediction))