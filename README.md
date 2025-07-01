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