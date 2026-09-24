import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ──────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Session State Initialization
# ──────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def go_single():
    st.session_state.page = "single"

def go_bulk():
    st.session_state.page = "bulk"

# ──────────────────────────────────────────────────────────────────────────────
# 3. Global CSS — Light Professional Theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp {
        background: #F8F9FC !important;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] span {
        color: #1F2937 !important;
    }

    /* Global headings */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: #111827 !important;
    }

    /* Button resets */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    /* Primary action buttons */
    .stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
    }
    .stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary / back buttons */
    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #4F46E5 !important;
        border: 1.5px solid #E0E7FF !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #EEF2FF !important;
        border-color: #C7D2FE !important;
    }

    /* Number inputs, selects, text inputs */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {
        background: #FFFFFF !important;
        border: 1.5px solid #D1D5DB !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        color: #1F2937 !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1.5px solid #D1D5DB !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] * {
        color: #1F2937 !important;
    }

    /* Slider text */
    [data-testid="stSlider"] p,
    [data-testid="stSlider"] span {
        color: #4F46E5 !important;
        font-weight: 600 !important;
    }

    /* Labels */
    label p, label span,
    [data-testid="stWidgetLabel"] label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Dataframe / table */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Stepper buttons */
    div[data-testid="stNumberInput"] button {
        background: #F3F4F6 !important;
        color: #4B5563 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: #E5E7EB !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }
    [data-testid="stFileUploader"] section {
        border: 2px dashed #D1D5DB !important;
        border-radius: 12px !important;
        background: #FAFBFD !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #FFFFFF !important;
        color: #059669 !important;
        border: 1.5px solid #A7F3D0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: #ECFDF5 !important;
        border-color: #6EE7B7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Load Machine Learning Model
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('best_churn_model.pkl')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Shared Prediction Helper
# ──────────────────────────────────────────────────────────────────────────────
RAW_FEATURE_COLUMNS = [
    'Customer_Type', 'City', 'Product_Category',
    'Purchase_Frequency', 'Total_Amount', 'Avg_Order_Value',
    'Last_Purchase_Days', 'Credit_Days', 'Payment_Mode',
    'Orders_Last_6_Months', 'Complaint_Count', 'Years_As_Customer'
]

def preprocess_and_predict(input_df):
    """
    Takes a DataFrame with the 12 raw feature columns,
    one-hot encodes categoricals, reindexes to the model's
    expected features, and returns (predictions, probabilities).
    """
    expected_features = getattr(model, 'feature_names_in_', None)
    if expected_features is not None:
        encoded_df = pd.get_dummies(input_df[RAW_FEATURE_COLUMNS])
        input_data = encoded_df.reindex(columns=expected_features, fill_value=0)
    else:
        input_data = input_df[RAW_FEATURE_COLUMNS]

    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1] * 100
    return predictions, probabilities

def risk_level(prob):
    if prob >= 70:
        return "HIGH"
    elif prob >= 30:
        return "MEDIUM"
    else:
        return "LOW"

def risk_color(level):
    return {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#059669"}.get(level, "#6B7280")

def risk_bg(level):
    return {"HIGH": "#FEF2F2", "MEDIUM": "#FFFBEB", "LOW": "#ECFDF5"}.get(level, "#F9FAFB")

def risk_icon(level):
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")


# ══════════════════════════════════════════════════════════════════════════════
#                              PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def render_home():

    # ── Hero Section ──
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 40%, #FDF2F8 100%);
        border: 1px solid #E0E7FF;
        border-radius: 20px;
        padding: 56px 40px 48px;
        text-align: center;
        margin-bottom: 40px;
    ">
        <div style="
            display: inline-flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
            width: 64px; height: 64px; border-radius: 16px;
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
            margin-bottom: 20px;
        ">
            <span style="font-size: 30px; line-height: 1;">📊</span>
        </div>
        <h1 style="
            font-size: 2.4rem; font-weight: 800;
            background: linear-gradient(135deg, #312E81, #4F46E5);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin: 0 0 8px;
        ">Customer Churn Prediction</h1>
        <p style="
            color: #6366F1; font-size: 1.15rem; font-weight: 500;
            margin: 0 0 16px;
        ">AI-powered customer risk analysis for smarter retention decisions.</p>
        <p style="
            color: #6B7280; font-size: 0.95rem; max-width: 620px;
            margin: 0 auto; line-height: 1.6;
        ">
            This application uses a trained Gradient Boosting machine learning model to predict
            whether a customer is likely to churn. Enter individual customer details or upload
            an entire dataset to generate churn risk scores and actionable insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Performance Cards ──
    st.markdown("""
    <p style="
        text-align: center; text-transform: uppercase; letter-spacing: 2px;
        font-size: 0.8rem; font-weight: 700; color: #6366F1; margin-bottom: 4px;
    ">Model Performance</p>
    <h2 style="text-align: center; font-size: 1.4rem; margin-bottom: 24px; color: #111827 !important;">
        Trained & Evaluated on Real Customer Data
    </h2>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    card_style = """
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    """
    label_style = "font-size: 0.78rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 6px;"
    value_style = "font-size: 1.6rem; font-weight: 800; color: #111827; margin: 4px 0 6px;"

    with m1:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="{label_style}">Model</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #4F46E5; margin: 6px 0;">Gradient Boosting</div>
            <div style="display:inline-block; background:#EEF2FF; color:#4F46E5; font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:50px;">Best Performer</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="{label_style}">Accuracy</div>
            <div style="{value_style}">87.9%</div>
            <div style="display:inline-block; background:#ECFDF5; color:#059669; font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:50px;">F1-Score: 0.85</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="{label_style}">Dataset</div>
            <div style="{value_style}">5,000+</div>
            <div style="display:inline-block; background:#F5F3FF; color:#7C3AED; font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:50px;">Customer Records</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="{label_style}">Task</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #111827; margin: 6px 0;">Binary Classification</div>
            <div style="display:inline-block; background:#FEF3C7; color:#D97706; font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:50px;">Precision: 0.86 · Recall: 0.84</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    # ── Choose Your Prediction Mode ──
    st.markdown("""
    <p style="
        text-align: center; text-transform: uppercase; letter-spacing: 2px;
        font-size: 0.8rem; font-weight: 700; color: #6366F1; margin-bottom: 4px;
    ">Choose Your Prediction Mode</p>
    <h2 style="text-align: center; font-size: 1.4rem; color: #111827 !important; margin-bottom: 4px;">
        Analyze One Customer or Evaluate an Entire Dataset
    </h2>
    <p style="text-align: center; color: #9CA3AF; font-size: 0.9rem; margin-bottom: 28px;">
        Select the mode that best fits your analysis needs.
    </p>
    """, unsafe_allow_html=True)

    col_s, col_b = st.columns(2, gap="large")

    with col_s:
        st.markdown("""
        <div style="
            background: #FFFFFF;
            border: 1.5px solid #E0E7FF;
            border-radius: 18px;
            padding: 36px 28px 28px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.03);
            min-height: 260px;
            display: flex; flex-direction: column; justify-content: space-between;
        ">
            <div>
                <div style="font-size: 40px; margin-bottom: 12px;">👤</div>
                <h3 style="font-size: 1.2rem; font-weight: 700; color: #111827 !important; margin: 0 0 8px;">Single Customer</h3>
                <p style="color: #6B7280; font-size: 0.9rem; line-height: 1.5; margin: 0 0 20px;">
                    Enter individual customer details and instantly predict their churn risk.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Predict Single Customer →", key="btn_single", on_click=go_single, type="primary", use_container_width=True)

    with col_b:
        st.markdown("""
        <div style="
            background: #FFFFFF;
            border: 1.5px solid #E0E7FF;
            border-radius: 18px;
            padding: 36px 28px 28px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.03);
            min-height: 260px;
            display: flex; flex-direction: column; justify-content: space-between;
        ">
            <div>
                <div style="font-size: 40px; margin-bottom: 12px;">📂</div>
                <h3 style="font-size: 1.2rem; font-weight: 700; color: #111827 !important; margin: 0 0 8px;">Bulk CSV Analysis</h3>
                <p style="color: #6B7280; font-size: 0.9rem; line-height: 1.5; margin: 0 0 20px;">
                    Upload a CSV containing multiple customers and generate churn predictions for the entire dataset.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Analyze CSV →", key="btn_bulk", on_click=go_bulk, type="primary", use_container_width=True)

    st.markdown("<div style='height: 48px'></div>", unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <p style="
        text-align: center; text-transform: uppercase; letter-spacing: 2px;
        font-size: 0.8rem; font-weight: 700; color: #6366F1; margin-bottom: 4px;
    ">How It Works</p>
    <h2 style="text-align: center; font-size: 1.4rem; color: #111827 !important; margin-bottom: 32px;">
        From Raw Data to Actionable Risk Scores
    </h2>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    step_card = """
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 24px 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        min-height: 140px;
    """
    step_num = "display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:50%;background:#EEF2FF;color:#4F46E5;font-weight:800;font-size:0.85rem;margin-bottom:10px;"
    step_title = "font-size:0.9rem;font-weight:700;color:#111827;margin:0 0 4px;"
    step_desc = "font-size:0.78rem;color:#9CA3AF;margin:0;"

    steps = [
        ("1", "Customer Data", "Raw feature inputs collected"),
        ("2", "Data Preprocessing", "Encoding & feature alignment"),
        ("3", "ML Model", "Gradient Boosting Classifier"),
        ("4", "Churn Probability", "Confidence score generated"),
        ("5", "Risk Classification", "LOW · MEDIUM · HIGH"),
    ]
    for col, (num, title, desc) in zip([s1, s2, s3, s4, s5], steps):
        with col:
            st.markdown(f"""
            <div style="{step_card}">
                <div style="{step_num}">{num}</div>
                <div style="{step_title}">{title}</div>
                <div style="{step_desc}">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div style="text-align:center; margin-top:56px; padding:20px 0; border-top:1px solid #E5E7EB;">
        <p style="color:#9CA3AF; font-size:0.8rem; margin:0;">
            Built with Streamlit · Gradient Boosting · Scikit-Learn
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#                        PAGE: SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
def render_single():

    # ── Back button ──
    st.button("← Back to Home", key="back_single", on_click=go_home, type="secondary")

    # ── Page header ──
    st.markdown("""
    <div style="margin-bottom: 28px;">
        <h1 style="font-size: 1.8rem; font-weight: 800; color: #111827 !important; margin: 0 0 4px;">
            👤 Single Customer Churn Prediction
        </h1>
        <p style="color: #6B7280; font-size: 0.95rem; margin: 0;">
            Enter customer details below and predict their individual churn risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Could not load model: {model_error}")
        return

    # ── Input Form ──
    st.markdown("""
    <div style="
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
        padding: 28px 24px 8px; margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    ">
        <h3 style="font-size: 1.05rem; font-weight: 700; color: #111827 !important; margin: 0 0 4px;">
            Customer Information
        </h3>
        <p style="color: #9CA3AF; font-size: 0.85rem; margin: 0 0 16px;">
            Provide the customer's profile and transactional details.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        customer_type = st.selectbox("Customer Type", ["Retailer", "Dealer", "Industry", "Contractor"], key="s_ctype")
        purchase_freq = st.slider("Purchase Frequency", 1, 50, 15, key="s_pf")
        total_amount = st.number_input("Total Spend Amount (₹)", 1000, 2000000, 250000, step=10000, key="s_ta")
        avg_order_value = st.number_input("Avg Order Value (₹)", 500, 100000, 16000, step=1000, key="s_aov")

    with col_b:
        city = st.selectbox("City", ["Chandigarh", "Ghaziabad", "Lucknow", "Noida", "Jaipur", "Agra", "Delhi", "Gurgaon", "Faridabad", "Kanpur"], key="s_city")
        last_purchase_days = st.slider("Last Purchase (Days Ago)", 1, 365, 45, key="s_lpd")
        orders_last_6m = st.slider("Orders (Last 6 Months)", 0, 30, 10, key="s_ol6")
        years_as_customer = st.slider("Years as Customer", 1, 20, 4, key="s_yac")

    with col_c:
        product_category = st.selectbox("Product Category", ["Tools", "Fasteners", "Bearings", "Chemicals", "Electrical", "Industrial Supplies", "Safety Equipment", "Lubricants"], key="s_pc")
        payment_mode = st.selectbox("Payment Mode", ["Cash", "Credit", "UPI", "Online", "Bank Transfer"], key="s_pm")
        credit_days = st.selectbox("Credit Days Allowed", [0, 15, 30, 45, 60, 90], index=3, key="s_cd")
        complaint_count = st.slider("Complaint Count", 0, 10, 1, key="s_cc")

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    btn_predict = st.button("🔮 Run Churn Prediction", key="btn_predict_single", type="primary", use_container_width=True)

    if btn_predict:
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
        predictions, probabilities = preprocess_and_predict(input_df)
        prediction = predictions[0]
        probability = probabilities[0]
        level = risk_level(probability)

        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        # ── Result Cards ──
        r1, r2, r3 = st.columns(3)

        with r1:
            if prediction == 1:
                st.markdown(f"""
                <div style="
                    background: #FEF2F2; border: 1.5px solid #FECACA; border-radius: 16px;
                    padding: 28px 20px; text-align: center;
                ">
                    <div style="font-size: 0.78rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 8px;">Prediction Result</div>
                    <div style="font-size: 28px; margin-bottom: 6px;">⚠️</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #DC2626;">High Churn Risk</div>
                    <p style="color: #6B7280; font-size: 0.82rem; margin: 8px 0 0;">Customer is likely to churn.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: #ECFDF5; border: 1.5px solid #A7F3D0; border-radius: 16px;
                    padding: 28px 20px; text-align: center;
                ">
                    <div style="font-size: 0.78rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 8px;">Prediction Result</div>
                    <div style="font-size: 28px; margin-bottom: 6px;">✅</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #059669;">Customer Retained</div>
                    <p style="color: #6B7280; font-size: 0.82rem; margin: 8px 0 0;">Customer is unlikely to churn.</p>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            prob_color = risk_color(level)
            st.markdown(f"""
            <div style="
                background: #FFFFFF; border: 1.5px solid #E5E7EB; border-radius: 16px;
                padding: 28px 20px; text-align: center;
            ">
                <div style="font-size: 0.78rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 12px;">Churn Probability</div>
                <div style="font-size: 2.4rem; font-weight: 800; color: {prob_color};">{probability:.1f}%</div>
                <p style="color: #9CA3AF; font-size: 0.82rem; margin: 8px 0 0;">Model confidence score</p>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div style="
                background: {risk_bg(level)}; border: 1.5px solid #E5E7EB; border-radius: 16px;
                padding: 28px 20px; text-align: center;
            ">
                <div style="font-size: 0.78rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 12px;">Risk Level</div>
                <div style="font-size: 2.4rem; font-weight: 800; color: {risk_color(level)};">{level}</div>
                <div style="display:inline-block; margin-top:8px; background:{risk_bg(level)}; color:{risk_color(level)}; font-size:0.75rem; font-weight:700; padding:3px 12px; border-radius:50px; border:1px solid {risk_color(level)}30;">
                    {risk_icon(level)} {level} Risk
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#                          PAGE: BULK CSV
# ══════════════════════════════════════════════════════════════════════════════
def render_bulk():

    # ── Back button ──
    st.button("← Back to Home", key="back_bulk", on_click=go_home, type="secondary")

    # ── Page header ──
    st.markdown("""
    <div style="margin-bottom: 28px;">
        <h1 style="font-size: 1.8rem; font-weight: 800; color: #111827 !important; margin: 0 0 4px;">
            📂 Bulk Customer Churn Analysis
        </h1>
        <p style="color: #6B7280; font-size: 0.95rem; margin: 0;">
            Upload a CSV file with customer data to generate churn predictions for the entire dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Could not load model: {model_error}")
        return

    # ── CSV Template Download ──
    template_df = pd.DataFrame(columns=RAW_FEATURE_COLUMNS)
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV Template",
        data=template_csv,
        file_name="churn_prediction_template.csv",
        mime="text/csv",
        key="dl_template"
    )

    st.markdown("""
    <p style="color: #9CA3AF; font-size: 0.82rem; margin-top: -8px; margin-bottom: 20px;">
        Download the template above, fill in your customer data, and upload it below.
    </p>
    """, unsafe_allow_html=True)

    # ── File Uploader ──
    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        key="csv_upload",
        help="CSV must contain these columns: " + ", ".join(RAW_FEATURE_COLUMNS)
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV file: {e}")
            return

        # ── Column Validation ──
        missing_cols = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            st.markdown(f"""
            <div style="
                background: #FEF2F2; border: 1.5px solid #FECACA; border-radius: 14px;
                padding: 20px 24px; margin: 16px 0;
            ">
                <h4 style="color: #DC2626 !important; font-size: 1rem; margin: 0 0 8px;">
                    ❌ Missing Required Columns
                </h4>
                <p style="color: #6B7280; font-size: 0.88rem; margin: 0 0 12px;">
                    The uploaded CSV is missing the following columns required by the model:
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                    {"".join(f'<span style="background:#FEE2E2; color:#DC2626; font-size:0.8rem; font-weight:600; padding:4px 12px; border-radius:50px; border:1px solid #FECACA;">{c}</span>' for c in missing_cols)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # ── Upload Summary ──
        st.markdown(f"""
        <div style="
            background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
            padding: 20px 24px; margin: 16px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        ">
            <h4 style="color: #111827 !important; font-size: 1rem; margin: 0 0 8px;">📄 File Uploaded Successfully</h4>
            <p style="color: #6B7280; font-size: 0.88rem; margin: 0;">
                <strong>File:</strong> {uploaded_file.name} &nbsp;·&nbsp;
                <strong>Customers:</strong> {len(df):,}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Data Preview**")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        btn_bulk_predict = st.button("🚀 Predict All Customers", key="btn_bulk_predict", type="primary", use_container_width=True)

        if btn_bulk_predict:
            with st.spinner("Running predictions..."):
                predictions, probabilities = preprocess_and_predict(df)

            df_result = df.copy()
            df_result['Churn_Prediction'] = predictions
            df_result['Churn_Probability'] = np.round(probabilities, 2)
            df_result['Risk_Level'] = df_result['Churn_Probability'].apply(risk_level)

            # ── Summary Cards ──
            total = len(df_result)
            high = int((df_result['Risk_Level'] == 'HIGH').sum())
            medium = int((df_result['Risk_Level'] == 'MEDIUM').sum())
            low = int((df_result['Risk_Level'] == 'LOW').sum())

            st.markdown("""
            <p style="
                text-transform: uppercase; letter-spacing: 2px;
                font-size: 0.78rem; font-weight: 700; color: #6366F1; margin: 24px 0 4px;
            ">Prediction Summary</p>
            """, unsafe_allow_html=True)

            sc1, sc2, sc3, sc4 = st.columns(4)
            summary_card = "background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px; padding:20px 16px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.04);"

            with sc1:
                st.markdown(f"""
                <div style="{summary_card}">
                    <div style="font-size: 24px; margin-bottom: 4px;">👥</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #111827;">{total:,}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; font-weight: 600;">Total Customers</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div style="{summary_card}">
                    <div style="font-size: 24px; margin-bottom: 4px;">🔴</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #DC2626;">{high:,}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; font-weight: 600;">High Risk</div>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div style="{summary_card}">
                    <div style="font-size: 24px; margin-bottom: 4px;">🟡</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #D97706;">{medium:,}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; font-weight: 600;">Medium Risk</div>
                </div>
                """, unsafe_allow_html=True)
            with sc4:
                st.markdown(f"""
                <div style="{summary_card}">
                    <div style="font-size: 24px; margin-bottom: 4px;">🟢</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #059669;">{low:,}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; font-weight: 600;">Low Risk</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

            # ── Prediction Table ──
            st.markdown("**Prediction Results**")

            display_df = df_result.copy()
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Churn_Prediction": st.column_config.NumberColumn("Churn Prediction", format="%d"),
                    "Churn_Probability": st.column_config.NumberColumn("Churn Probability (%)", format="%.2f"),
                    "Risk_Level": st.column_config.TextColumn("Risk Level"),
                },
            )

            # ── Download Results ──
            csv_buffer = io.StringIO()
            df_result.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Predictions",
                data=csv_buffer.getvalue(),
                file_name="churn_predictions.csv",
                mime="text/csv",
                key="dl_predictions"
            )


# ══════════════════════════════════════════════════════════════════════════════
#                              ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "single":
    render_single()
elif st.session_state.page == "bulk":
    render_bulk()