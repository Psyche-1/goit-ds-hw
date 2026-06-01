import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os


CLASS_NAMES = ["Футболка", "Штани", "Пуловер", "Сукня", "Пальто",
               "Сандалі", "Сорочка", "Кросівки", "Сумка", "Черевики"]

st.title("Neural Network Visualization App")
st.write("Upload an image to classify it using one of the trained models.")

model_option = st.sidebar.selectbox(
    'Select a model for classification:',
    ('Convolutional Neural Network', 'VGG-based Model')
)

if model_option == 'Convolutional Neural Network':
    model_path_keras = 'model_cnn.keras'
    metrics_path = 'cnn_metrics.png'
else:
    model_path_keras = 'model_vgg.keras'
    metrics_path = 'vgg_metrics.png'

st.subheader("Model Training Metrics")
if os.path.exists(metrics_path):
    st.image(metrics_path, use_container_width=True)
else:
    st.warning(f"Metrics plot not found at: {metrics_path}")

invert_colors = st.sidebar.checkbox("Invert image colors (recommended for dark clothing on white background)", value=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)

    if st.button("Classify Image"):
        if not os.path.exists(model_path_keras):
            st.error(f"Model file not found at: {model_path_keras}. Please check Google Drive.")
        else:
            with st.spinner("Loading model and predicting... This may take a few seconds."):
                try:
                    model = tf.keras.models.load_model(model_path_keras)

                    input_shape = model.input_shape[1:3]

                    img_gray = image.convert('L')

                    if invert_colors:
                        img_gray = ImageOps.invert(img_gray)

                    img_resized = img_gray.resize(input_shape)
                    img_array = np.array(img_resized)

                    if len(img_array.shape) == 2:
                        img_array = np.expand_dims(img_array, axis=-1)
                    elif img_array.shape[-1] != 1:
                        img_array = img_array[:, :, :1]

                    img_array = img_array.astype('float32') / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array)[0]
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class = CLASS_NAMES[predicted_class_idx]

                    st.success(f"**Predicted Class:** {predicted_class}")

                    st.subheader("Class Probabilities:")
                    for i, prob in enumerate(predictions):
                        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
                        st.write(f"- **{class_name}**: {prob*100:.2f}%")
                        st.progress(float(prob))

                    del model
                    tf.keras.backend.clear_session()

                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")