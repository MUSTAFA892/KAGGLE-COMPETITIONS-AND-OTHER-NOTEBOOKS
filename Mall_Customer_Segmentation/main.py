import gradio as gr
import pickle
import numpy as np

# Load saved model and scaler
kmeans = pickle.load(open("model/kmeans_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))    

# Mapping gender input
gender_map = {"Female": 0, "Male": 1}

# Prediction function
def predict_cluster(gender, age, income, score):
    gender_encoded = gender_map[gender]
    input_data = np.array([[gender_encoded, age, income, score]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    return f"Predicted Customer Cluster: {cluster}"

# Gradio UI
interface = gr.Interface(
    fn=predict_cluster,
    inputs=[
        gr.Radio(["Female", "Male"], label="Gender"),
        gr.Slider(18, 70, label="Age"),
        gr.Slider(15, 137, label="Annual Income (k$)"),
        gr.Slider(1, 99, label="Spending Score (1-100)")
    ],
    outputs=gr.Text(label="Cluster Prediction"),
    title="Customer Segmentation using KMeans",
    description="Input customer data to predict their cluster segment"
)

# Launch app
interface.launch()
