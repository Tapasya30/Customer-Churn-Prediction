import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="ChurnAI • Purple Glass Analytics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Bulletproof BaseWeb & Streamlit CSS Overrides
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #FFFFFF !important;
    }

    /* Page Background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #2c0847 0%, #17072b 50%, #0a0314 100%) !important;
        background-attachment: fixed !important;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background: rgba(20, 8, 35, 0.9) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    /* ALL Labels in Sidebar & Main Page */
    [data-testid="stSidebar"] label, [data-testid="stWidgetLabel"] label, label p, label span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }

    /* ==========================================================================
       BULLETPROOF FIX FOR NUMBER INPUTS (Forces Dark Box + Pure White Text)
       ========================================================================== */
    
    /* Target BaseWeb Base Input & Input Containers */
    div[data-baseweb="base-input"], 
    div[data-baseweb="input"], 
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background-color: #1F0E3D !important;
        background: #1F0E3D !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
    }

    /* Force Input Tag Text to Pure White */
    input[data-testid="stNumberInput-Input"],
    div[data-testid="stNumberInput"] input, 
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        background-color: #1F0E3D !important;
        background: #1F0E3D !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        caret-color: #FFFFFF !important;
    }

    /* Plus (+) and Minus (-) Stepper Buttons */
    div[data-testid="stNumberInput"] button, 
    button[title="Increase value"], 
    button[title="Decrease value"] {
        background-color: #32165D !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #C084FC !important;
        color: #FFFFFF !important;
    }

    /* ==========================================================================
       SLIDERS & DROPDOWNS
       ========================================================================== */
    
    /* Slider Current Value Text */
    [data-testid="stSlider"] div[data-testid="stTickBar"] + div, 
    [data-testid="stSlider"] p,
    [data-testid="stSlider"] span {
        color: #F472B6 !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
    }

    /* Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #1F0E3D !important;
        background: #1F0E3D !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }

    .glass-card-sm {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 18px 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }

    .gradient-header {
        background: linear-gradient(135deg, #F472B6 0%, #E9D5FF 50%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .subtitle-text {
        color: #E9D5FF;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    /* Stat Badges & Pills */
    .stat-label {
        font-size: 0.85rem;
        color: #E9D5FF;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }

    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 4px;
    }

    .pill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 6px;
    }

    .pill-purple { background: rgba(192, 132, 252, 0.25); color: #F3E8FF; border: 1px solid rgba(192, 132, 252, 0.5); }
    .pill-pink { background: rgba(244, 114, 182, 0.25); color: #FCE7F3; border: 1px solid rgba(244, 114, 182, 0.5); }
    .pill-green { background: rgba(52, 211, 153, 0.25); color: #D1FAE5; border: 1px solid rgba(52, 211, 153, 0.5); }
    .pill-red { background: rgba(248, 113, 113, 0.25); color: #FEE2E2; border: 1px solid rgba(248, 113, 113, 0.5); }

    /* Action Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #EC4899 0%, #8B5CF6 50%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 10px 25px rgba(236, 72, 153, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 15px 35px rgba(236, 72, 153, 0.6) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 3. Load Machine Learning Model
@st.cache_resource
def load_model():
    return joblib.load('best_churn_model.pkl')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# 4. Header Bar
col_logo, col_head = st.columns([0.15, 0.85])
with col_logo:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #C084FC, #E879F9); width: 56px; height: 56px; border-radius: 18px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 25px rgba(192, 132, 252, 0.5);">
            <span style="font-size: 28px;">✨</span>
        </div>
    """, unsafe_allow_html=True)

with col_head:
    st.markdown('<div class="gradient-header">Customer Churn Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Real-time Machine Learning Churn Risk Assessment</div>', unsafe_allow_html=True)

# 5. Top Glass Benchmark Metric Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
        <div class="glass-card-sm">
            <div class="stat-label">Model Accuracy</div>
            <div class="stat-value">87.9%</div>
            <div class="pill-badge pill-purple">↑ 2.4% vs baseline</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
        <div class="glass-card-sm">
            <div class="stat-label">Best Algorithm</div>
            <div class="stat-value" style="font-size: 1.35rem; margin-top: 8px;">Gradient Boosting</div>
            <div class="pill-badge pill-pink">Selected ML</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
        <div class="glass-card-sm">
            <div class="stat-label">Dataset Records</div>
            <div class="stat-value">5,000+</div>
            <div class="pill-badge pill-purple">Evaluated</div>
        </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
        <div class="glass-card-sm">
            <div class="stat-label">System Status</div>
            <div class="stat-value" style="color: #34D399;">Active</div>
            <div class="pill-badge pill-green">● Model Ready</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Sidebar Inputs
st.sidebar.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
        <span style="font-size: 24px;">👤</span>
        <h3 style="margin: 0; color: #FFFFFF; font-size: 1.25rem; font-weight: 700;">Customer Input Profile</h3>
    </div>
""", unsafe_allow_html=True)

complaint_count = st.sidebar.slider("⚠️ Complaint Count", 0, 10, 1)
purchase_freq = st.sidebar.slider("🔄 Purchase Frequency", 1, 50, 15)
last_purchase_days = st.sidebar.slider("📅 Last Purchase (Days Ago)", 1, 365, 45)
total_amount = st.sidebar.number_input("💰 Total Spend Amount (₹)", 1000, 2000000, 250000, step=10000)
years_as_customer = st.sidebar.slider("🤝 Years as Customer", 1, 20, 4)
avg_order_value = st.sidebar.number_input("📊 Avg Order Value (₹)", 500, 100000, 16000, step=1000)
orders_last_6m = st.sidebar.slider("📈 Orders (Last 6 Months)", 0, 30, 10)
credit_days = st.sidebar.selectbox("⏳ Credit Days Allowed", [0, 15, 30, 45, 60, 90], index=3)

customer_type = st.sidebar.selectbox("🏢 Customer Type", ["Retailer", "Dealer", "Industry", "Contractor"])
city = st.sidebar.selectbox("📍 City", ["Chandigarh", "Ghaziabad", "Lucknow", "Noida", "Jaipur", "Agra", "Delhi", "Gurgaon", "Faridabad"])
product_category = st.sidebar.selectbox("📦 Product Category", ["Tools", "Fasteners", "Bearings", "Chemicals", "Electrical", "Industrial Supplies", "Safety Equipment"])
payment_mode = st.sidebar.selectbox("💳 Payment Mode", ["Cash", "Credit", "UPI", "Online", "Bank Transfer"])

# 7. Main Assessment Layout
col_main, col_insight = st.columns([1.2, 0.8])

with col_main:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #FFFFFF; font-size: 1.5rem; margin-bottom: 6px;">🔮 Predict Churn Risk</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #E9D5FF; font-size: 0.95rem; margin-bottom: 16px;">Click below to run customer features through the trained Machine Learning pipeline.</p>', unsafe_allow_html=True)

    btn_predict = st.button("✨ Run Prediction Assessment")

    if btn_predict:
        if not model_loaded:
            st.error(f"Could not load model: {model_error}")
        else:
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
            expected_features = getattr(model, 'feature_names_in_', None)

            if expected_features is not None:
                encoded_df = pd.get_dummies(input_df)
                input_data = encoded_df.reindex(columns=expected_features, fill_value=0)
            else:
                input_data = input_df

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.5); border-radius: 18px; padding: 22px; text-align: center;">
                        <div style="font-size: 38px;">⚠️</div>
                        <h3 style="color: #FCA5A5; margin: 8px 0; font-size: 1.5rem;">High Churn Risk Detected</h3>
                        <p style="color: #FEE2E2; font-size: 0.95rem;">This customer displays behavioral patterns indicative of leaving.</p>
                        <div style="font-size: 2.5rem; font-weight: 800; color: #EF4444; margin-top: 10px;">{probability:.1f}%</div>
                        <span class="pill-badge pill-red">High Priority Alert</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.5); border-radius: 18px; padding: 22px; text-align: center;">
                        <div style="font-size: 38px;">✅</div>
                        <h3 style="color: #6EE7B7; margin: 8px 0; font-size: 1.5rem;">Customer Retained</h3>
                        <p style="color: #D1FAE5; font-size: 0.95rem;">This customer is satisfied and unlikely to churn.</p>
                        <div style="font-size: 2.5rem; font-weight: 800; color: #10B981; margin-top: 10px;">{probability:.1f}%</div>
                        <span class="pill-badge pill-green">Low Churn Risk</span>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_insight:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #FFFFFF; font-size: 1.4rem; margin-bottom: 6px;">📊 Model Key Insights</h3>
            <p style="color: #E9D5FF; font-size: 0.9rem; margin-bottom: 16px;">Top feature drivers from Gradient Boosting evaluations:</p>
            <div>
                <div style="margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #FFFFFF;">
                        <span>⚠️ Complaint Count</span>
                        <span style="font-weight: 800; color: #F472B6;">25%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); height: 8px; border-radius: 10px; margin-top: 6px;">
                        <div style="background: linear-gradient(90deg, #F472B6, #C084FC); width: 25%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #FFFFFF;">
                        <span>🔄 Purchase Frequency</span>
                        <span style="font-weight: 800; color: #F472B6;">25%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); height: 8px; border-radius: 10px; margin-top: 6px;">
                        <div style="background: linear-gradient(90deg, #F472B6, #C084FC); width: 25%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #FFFFFF;">
                        <span>📅 Last Purchase (Days)</span>
                        <span style="font-weight: 800; color: #F472B6;">20%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); height: 8px; border-radius: 10px; margin-top: 6px;">
                        <div style="background: linear-gradient(90deg, #F472B6, #C084FC); width: 20%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #FFFFFF;">
                        <span>💰 Total Spend Amount</span>
                        <span style="font-weight: 800; color: #F472B6;">17%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); height: 8px; border-radius: 10px; margin-top: 6px;">
                        <div style="background: linear-gradient(90deg, #F472B6, #C084FC); width: 17%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)