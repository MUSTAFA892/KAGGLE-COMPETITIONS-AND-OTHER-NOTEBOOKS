import gradio as gr
import numpy as np
from PIL import Image

def predict_cat_dog(img):
    # Resize and preprocess
    img = img.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    if prediction > 0.5:
        label = "🐶 Dog"
    else:
        label = "🐱 Cat"

    return {label: float(confidence)}

demo = gr.Interface(
    fn=predict_cat_dog,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Cat vs Dog Classifier 🐱🐶",
    description="Upload a cat/dog image to classify."
)

demo.launch(share=True)