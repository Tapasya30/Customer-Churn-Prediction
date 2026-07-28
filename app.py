import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="wide")

st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("Adjust customer attributes on the left and click **Predict Churn Risk** to get predictions.")

# Load Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load('best_churn_model.pkl')

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Get the exact feature names expected by the model
expected_features = getattr(model, 'feature_names_in_', None)

# User Inputs
st.sidebar.header("Customer Profile Input")

# 1. Main Numerical Inputs
complaint_count = st.sidebar.slider("Complaint Count", 0, 10, 1)
purchase_freq = st.sidebar.slider("Purchase Frequency", 1, 50, 15)
last_purchase_days = st.sidebar.slider("Last Purchase (Days Ago)", 1, 365, 45)
total_amount = st.sidebar.number_input("Total Amount Spent (₹)", 1000, 2000000, 250000)
years_as_customer = st.sidebar.slider("Years As Customer", 1, 20, 4)
avg_order_value = st.sidebar.number_input("Avg Order Value (₹)", 500, 100000, 16000)
orders_last_6m = st.sidebar.slider("Orders in Last 6 Months", 0, 30, 10)
credit_days = st.sidebar.selectbox("Credit Days Allowed", [0, 15, 30, 45, 60, 90])

# 2. Main Categorical Inputs
customer_type = st.sidebar.selectbox("Customer Type", ["Retailer", "Dealer", "Industry", "Contractor"])
city = st.sidebar.selectbox("City", ["Chandigarh", "Ghaziabad", "Lucknow", "Noida", "Jaipur", "Agra", "Delhi", "Gurgaon", "Faridabad"])
product_category = st.sidebar.selectbox("Product Category", ["Tools", "Fasteners", "Bearings", "Chemicals", "Electrical", "Industrial Supplies", "Safety Equipment"])
payment_mode = st.sidebar.selectbox("Payment Mode", ["Cash", "Credit", "UPI", "Online", "Bank Transfer"])

if st.button("Predict Churn Risk", type="primary"):
    # Raw input dictionary
    raw_input = {
        'Customer_Type': customer_type,
        'City': city,
        'Product_Category': product_category,
        'Purchase_Frequency': purchase_freq,
        'Total_Amount': total_amount,
        'Avg_Order_Value': avg_order_value,
        'Last_Purchase_Days': last_purchase_days,
        'Credit_Days': credit_days,
        'Payment_Mode': payment_mode,
        'Orders_Last_6_Months': orders_last_6m,
        'Complaint_Count': complaint_count,
        'Years_As_Customer': years_as_customer
    }
    
    input_df = pd.DataFrame([raw_input])
    
    # If model was trained directly on One-Hot Encoded features (without Pipeline):
    if expected_features is not None:
        # Perform One-Hot Encoding on categorical features to match model training features
        encoded_df = pd.get_dummies(input_df)
        
        # Reindex dataframe so all expected columns are present, filled with 0 if missing
        input_data = encoded_df.reindex(columns=expected_features, fill_value=0)
    else:
        input_data = input_df

    # Make Prediction
    try:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        st.write("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.error(f"⚠️ **High Churn Risk!**")
            else:
                st.success(f"✅ **Low Churn Risk**")
        with col2:
            st.metric(label="Churn Probability", value=f"{prob * 100:.2f}%")
            
    except Exception as err:
        st.error(f"Prediction Error: {err}")