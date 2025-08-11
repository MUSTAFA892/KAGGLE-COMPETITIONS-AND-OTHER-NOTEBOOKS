import pandas as pd
import numpy as np
import gradio as gr
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# ----------------------------
# Data loading & preprocessing
# ----------------------------
df = pd.read_csv("Datasets/survey_results_public.csv")
df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
df = df.rename({"ConvertedComp": "Salary"}, axis=1)
df = df[df["Salary"].notnull()]
df = df.dropna()
df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)

# Shorten country categories
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 10000]
df = df[df['Country'] != 'Other']

# Clean YearsCodePro
def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

# Clean EdLevel
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_education)

# Encode categorical data
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])

le_edlevel = LabelEncoder()
df['EdLevel'] = le_edlevel.fit_transform(df['EdLevel'])

# ----------------------------
# Model training
# ----------------------------
X = df.drop("Salary", axis=1)
y = df["Salary"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Prediction function for Gradio
# ----------------------------
def predict_salary(country, edlevel, yearscode):
    try:
        country_enc = le_country.transform([country])[0]
        edlevel_enc = le_edlevel.transform([edlevel])[0]
        X_input = np.array([[country_enc, edlevel_enc, float(yearscode)]])
        prediction = model.predict(X_input)[0]
        return f"Estimated Salary: ${prediction:,.2f} USD"
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------
# Gradio Interface
# ----------------------------
country_options = list(le_country.classes_)
edlevel_options = list(le_edlevel.classes_)

iface = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Dropdown(choices=country_options, label="Country"),
        gr.Dropdown(choices=edlevel_options, label="Education Level"),
        gr.Number(label="Years of Professional Coding Experience")
    ],
    outputs=gr.Textbox(label="Predicted Salary"),
    title="Developer Salary Prediction",
    description="Predict salary based on Country, Education Level, and Years of Coding Experience."
)

if __name__ == "__main__":
    iface.launch()
