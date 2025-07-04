# 🏦 Bank Loan Classification

This project implements an end-to-end machine learning pipeline to predict whether a customer will opt for a personal loan based on demographic and financial attributes. It includes detailed EDA, feature engineering, preprocessing pipelines, model tuning, and model tracking using MLflow.

---

---

## 📊 Features Used

- **Demographic**: Age, Experience, Education, Family
- **Financial**: Income, Mortgage, CCAvg
- **Behavioral**: Online, CreditCard
- **Target**: `Personal Loan` (0/1)

---

## 🧪 Workflow Summary

### 1. **Exploratory Data Analysis (EDA)**
- Visualize target distribution, numeric histograms, boxplots.
- Pairplots, correlation matrix.
- Categorical feature vs target (Education, Family, Online, CreditCard).

### 2. **Preprocessing**
- Skewness correction using `PowerTransformer`.
- Scaling using `RobustScaler` and `StandardScaler`.
- Transformers saved using `joblib`.

### 3. **Feature Selection**
- Recursive Feature Elimination (RFE) using Logistic Regression.

### 4. **Handling Imbalance**
- SMOTE applied to oversample the minority class.

### 5. **Model Training & Tuning**
- GridSearchCV used for hyperparameter tuning on:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors
  - Support Vector Machine

### 6. **MLflow Tracking**
- Metrics: Accuracy, Precision, Recall, F1-score
- Models are logged and the best one is:
  - Registered under `BankLoanBestModel`
  - Promoted to **Production** stage

---

## 🚀 How to Run

### 1. ✅ Install dependencies:
### pip install -r requirements.txt

### ✅ Run the Jupyter Notebook
Open and execute the Assignment.ipynb file for exploratory data analysis and preprocessing.

### ✅ Execute the Application Scripts
### In your terminal or command prompt, run the following:

python server.py
python client.py


## For Web Interface
### Execute following command for flask based ui:
python Flask_app.py

### Execute following command for streamlit based ui:
streamlit run Streamlit_main.py


📁 Screenshots Directory Guide

This folder contains categorized screenshots related to various stages of your ML pipeline project.

────────────────────────────────────────────

📂 EDA
• File:
   - Eda_and_DataProcessing.ipynb
• Description:
   - This notebook outlines Exploratory Data Analysis and preprocessing steps used in the pipeline.

────────────────────────────────────────────

📂 Flask
• Files:
   - image.png
   - Screenshot 2025-07-03 124940.png
• Description:
   - Shows the Flask app interface for file upload and predictions using trained models.

────────────────────────────────────────────

📂 MlFlow_and_Evidently
• Files:
   - Screenshot 2025-07-03 125028.png
   - Screenshot 2025-07-03 125108.png
   - Screenshot 2025-07-03 125134.png
• Description:
   - MLflow UI and Evidently AI integration snapshots demonstrating experiment tracking and drift analysis.

────────────────────────────────────────────

📂 Streamlit
• Files:
   - Screenshot 2025-07-03 125339.png
   - Screenshot 2025-07-03 125414.png
• Description:
   - Screenshots of the Streamlit interface built for user interaction and model prediction.

────────────────────────────────────────────


📂 ApacheAirflow
• Files:
   - Screenshot 2025-07-03 215816.png
   - Screenshot 2025-07-03 215826.png
   - Screenshot 2025-07-03 215837.png
   - Screenshot 2025-07-03 215929.png
   - Screenshot 2025-07-03 215950.png
   - Screenshot 2025-07-03 220021.png


────────────────────────────────────────────
