# 💼 Employee Salary Classification

Predict whether an employee earns **>$50K or ≤$50K** based on demographic and work features using machine learning.

---

## 📋 Project Overview

This project implements a **comprehensive machine learning pipeline** for binary classification of employee salaries using the **Adult Census Income dataset**. It compares multiple machine learning algorithms to predict whether an individual's income exceeds $50,000 based on demographic and employment-related features.

---

## 🎯 Key Features

- ✅ **Multi-Model Comparison**: Compares 5 different ML algorithms  
- 🌐 **Interactive Web App**: Streamlit interface for real-time predictions  
- 📂 **Batch Processing**: Upload CSV for bulk predictions  
- 💾 **Model Persistence**: Automatically saves the best model and artifacts  
- 📊 **Comprehensive Evaluation**: Performance metrics and model selection  

---

## 🔧 Technologies Used

- **Language**: Python 3.7+
- **ML Framework**: scikit-learn
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Deployment**: Streamlit Community Cloud

---

## 🤖 Machine Learning Models

- Logistic Regression  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Gradient Boosting Classifier  

---

## 📊 Dataset Information

**Source**: UCI Machine Learning Repository – [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
**Records**: 48,842  
**Features**: 14 input features  
**Target**: Binary classification — `<=50K` or `>50K`

### 🔍 Features Used

| Feature           | Description                  | Type        |
|-------------------|------------------------------|-------------|
| age               | Age in years                 | Numeric     |
| workclass         | Employment type              | Categorical |
| fnlwgt            | Final sampling weight        | Numeric     |
| educational-num   | Years of education           | Numeric     |
| marital-status    | Marital status               | Categorical |
| occupation        | Job type                     | Categorical |
| relationship      | Family relationship          | Categorical |
| race              | Race                         | Categorical |
| gender            | Gender                       | Categorical |
| capital-gain      | Capital gains                | Numeric     |
| capital-loss      | Capital losses               | Numeric     |
| hours-per-week    | Work hours per week          | Numeric     |
| native-country    | Country of origin            | Categorical |

---

## 🚀 Installation

### 📦 Prerequisites

- Python 3.7 or higher  
- pip package manager

### 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/employee-salary-classification.git
cd employee-salary-classification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

# Install dependencies
pip install -r requirements.txt ```

---

# 📁 Project Structure
employee-salary-classification/
│
├── models/                     # Saved model artifacts
│   ├── best_model.pkl          # Best-performing model
│   ├── scaler.pkl              # Feature scaler
│   └── feature_columns.pkl     # Feature names
│
├── model_training.py           # Model training pipeline
├── streamlit_app.py            # Web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── data/
    └── adult.csv               # Dataset file

---    

# 📈 Model Performance
| Algorithm           | Accuracy | Notes                      |
| ------------------- | -------- | -------------------------- |
| Gradient Boosting   | \~87.5%  | Usually best performer     |
| Random Forest       | \~86.8%  | Balanced speed & accuracy  |
| Logistic Regression | \~84.2%  | Fast and interpretable     |
| SVM                 | \~83.1%  | Handles complex boundaries |
| KNN                 | \~81.4%  | Simple baseline            |
  
  ⚠️ Results may vary based on preprocessing and parameters.

---

#🔮 Future Enhancements
- **🔍 Hyperparameter Tuning (GridSearchCV)**
- **🧠 Deep Learning models**
- **✨ Feature Engineering**
- **📊 Model Explainability (SHAP, LIME)** 
- **🔁 Real-time Model Retraining** 
- **🧪 Unit Testing and CI/CD**
- **🐳 Docker Support**
- **🔌 REST API Integration**

