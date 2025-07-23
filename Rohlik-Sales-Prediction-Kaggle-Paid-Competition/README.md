
---

# Rohlík Sales Prediction

This repository contains the code and resources for the Rohlík Sales Prediction competition. The goal of this project is to develop a machine learning model capable of predicting future sales for Rohlík, an e-commerce grocery company, based on historical sales data and various other factors.

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
- [Modeling](#modeling)
- [Results](#results)
- [License](#license)



## Project Overview  

The objective of this project is to predict future sales for Rohlík based on historical data. The dataset contains records of past sales, marketing activities, promotional events, holidays, and other influencing factors that impact sales. The goal is to use these features to build a predictive model that can forecast future sales and help Rohlík optimize their operations and inventory management.

## Datasets

The dataset provided for this competition contains historical sales data with the following key features:

- `sales_data.csv`: Historical sales data including product information, sales volume, marketing campaigns, and other relevant variables.
- `test_data.csv`: Test data that contains the input features for which predictions must be made.
- `sample_submission.csv`: A sample submission file to guide the format of your predictions.

Please refer to the `datasets/` folder for the actual dataset files.

## File Structure

The repository contains the following files and folders:

```
Rohlik-Sales-Prediction/
├── datasets/                 # Folder containing the dataset files
│   ├── sales_data.csv
│   ├── test_data.csv
│   └── sample_submission.csv
├── notebooks/                # Folder containing Jupyter Notebooks for analysis and model training
│   ├── exploratory_analysis.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
├── README.md                 # This file
└── requirements.txt          # Required libraries for running the code
```

### Jupyter Notebooks

- `exploratory_analysis.ipynb`: Exploratory data analysis (EDA) notebook for understanding the dataset, visualizing trends, and identifying important factors.
- `preprocessing.ipynb`: Data preprocessing steps including feature engineering, handling missing values, and scaling.
- `model_training.ipynb`: Training and evaluating various machine learning models to predict future sales.

## Getting Started

To get started with this project, clone the repository and install the required dependencies:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/Rohlik-Sales-Prediction.git
   cd Rohlik-Sales-Prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebooks from the `notebooks/` folder:

   - `exploratory_analysis.ipynb` for understanding the sales data and visualizing patterns.
   - `preprocessing.ipynb` for preparing the data for training.
   - `model_training.ipynb` for building, tuning, and evaluating models.

## Modeling

Various machine learning models are trained in the `model_training.ipynb` notebook, including:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Machines (GBM)
- XGBoost
- LightGBM, etc.

Hyperparameter tuning and cross-validation are applied to identify the best-performing model.

## Results

Model performance is evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Once the best model is selected, predictions are made on the test data and saved in the appropriate format for submission to the competition platform.

## License

This repository is licensed under the MIT License.
