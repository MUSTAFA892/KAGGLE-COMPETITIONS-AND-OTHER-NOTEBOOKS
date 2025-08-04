import gradio as gr
import joblib
import numpy as np

# Load the saved model
model = joblib.load("model/svm_model.pkl")  

def predict_heart_disease(age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, slope, ca, thal):
    features = np.array([[age, sex, cp, trestbps, chol, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]
    return "Disease Detected 💔" if prediction == 1 else "No Disease 💖"

inputs = [
    gr.Number(label="Age"),
    gr.Radio([0, 1], label="Sex (0: Female, 1: Male)"),
    gr.Radio([0, 1, 2, 3], label="Chest Pain Type (cp)"),
    gr.Number(label="Resting BP (trestbps)"),
    gr.Number(label="Cholesterol (chol)"),
    gr.Radio([0, 1, 2], label="RestECG"),
    gr.Number(label="Max Heart Rate (thalach)"),
    gr.Radio([0, 1], label="Exercise Induced Angina (exang)"),
    gr.Number(label="ST Depression (oldpeak)"),
    gr.Radio([0, 1, 2], label="Slope"),
    gr.Radio([0, 1, 2, 3], label="Major Vessels (ca)"),
    gr.Radio([0, 1, 2], label="Thal (0=Normal, 1=Fixed, 2=Reversible)")
]

gr.Interface(fn=predict_heart_disease, inputs=inputs, outputs="text",
             title="Heart Disease Predictor",
             description="Enter patient info to check for heart disease").launch()
