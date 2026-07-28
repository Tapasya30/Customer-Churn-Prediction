# 🔮 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning and Analytics solution designed to predict customer churn, evaluate key behavioral risk factors, and provide real-time risk scores through an interactive Glassmorphism web dashboard.

---

## 🌐 Live Interactive Web App

Experience the live deployed dashboard in action:

👉 **[🚀 Launch Customer Churn Predictor App](https://your-deployed-app-link.streamlit.app)**

*(Note: Replace the link above with your live Streamlit Cloud URL once deployed!)*

---

## 📌 Executive Summary

Customer churn is a critical business metric for B2B and retail enterprises. Retaining existing customers is significantly more cost-effective than acquiring new ones. 

This project analyzes over **5,000+ customer records** to train, compare, and fine-tune multiple Machine Learning algorithms. The optimal model—**Gradient Boosting Classifier**—achieves **87.9% Accuracy** and identifies key operational metrics (e.g., complaint counts, purchase recency) driving customer drop-offs.

---

## 🚀 Key Project Features

- **Multi-Algorithm Evaluation**: Evaluated 5 machine learning models including *Logistic Regression, Decision Tree, Random Forest, XGBoost, and Gradient Boosting*.
- **Data Pipeline & Encoding**: Built automated feature scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`).
- **Feature Importance Analysis**: Identified the top 4 core business metrics influencing customer retention.
- **Glassmorphism Web Dashboard**: Custom Streamlit application UI featuring a dark purple glass aesthetic, real-time risk probability calculation, and high-contrast metrics.
- **Production Pipeline Export**: Serialized model pipeline via `joblib` for seamless inference and deployment.

---

## 📊 Model Performance Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** 🏆 | **87.9%** | **0.86** | **0.84** | **0.85** |
| **Random Forest** | 86.4% | 0.84 | 0.83 | 0.83 |
| **XGBoost** | 85.8% | 0.83 | 0.82 | 0.82 |
| **Decision Tree** | 79.2% | 0.76 | 0.77 | 0.76 |
| **Logistic Regression** | 76.5% | 0.73 | 0.71 | 0.72 |

> **Selected Model**: **Gradient Boosting Classifier** achieved the highest overall F1-Score and ROC-AUC curve performance.

---

## 💡 Key Business Insights (Top Churn Drivers)

Based on feature importance evaluation from the Gradient Boosting model:

1. **Complaint Count (25%)**: Customers logging multiple complaints without quick resolution show the highest churn risk.
2. **Purchase Frequency (25%)**: A drop in order frequency is an immediate indicator of customer disengagement.
3. **Last Purchase Days (20%)**: Inactivity exceeding 90 days strongly correlates with account cancellation.
4. **Total Spend Amount (17%)**: High-value accounts require specialized retention incentives to prevent switching.

---

## 📁 Repository Structure

```text
Churn prediction/
├── customer_churn_prediction.ipynb  # Data EDA, preprocessing & model training notebook
├── best_churn_model.pkl             # Serialized best trained ML pipeline
├── app.py                            # Streamlit Glassmorphism Web App UI
├── requirements.txt                  # Python package dependencies
├── MPG_Customer_Churn_Dataset.csv    # Customer Churn Dataset
└── README.md                         # Project documentation
```

---

## 🛠️ Technology Stack

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, Gradient Boosting, Joblib
- **Data Processing**: Pandas, NumPy
- **Web App & UI**: Streamlit, Custom Glassmorphism CSS
- **Visualization**: Matplotlib, Seaborn


