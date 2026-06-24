import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Customer Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.stApp {
    background-color: #fafafa;
    background-image:
        radial-gradient(at 0% 0%, #fef3c7 0px, transparent 50%),
        radial-gradient(at 100% 0%, #dbeafe 0px, transparent 50%),
        radial-gradient(at 100% 100%, #e0e7ff 0px, transparent 50%),
        radial-gradient(at 0% 100%, #fce7f3 0px, transparent 50%);
    background-attachment: fixed;
    color: #0f172a;
}
div[data-baseweb="select"] * {
    color: #1e293b!important;
    font-weight: 500!important;
}
div[data-baseweb="select"] {
    background: white!important;
    border: 2px solid #dbeafe!important;
    border-radius: 12px!important;
}
ul[role="listbox"] li {
    background: white!important;
    color: #1e293b!important;
}
.stNumberInput input {
    color: #1e293b!important;
    background: white!important;
    font-weight: 500!important;
}
.stSlider p {
    color: #1e293b!important;
}
h1 {
    color: #1e293b!important;
    font-weight: 800!important;
    letter-spacing: -0.025em;
}
h2, h3, h4 {
    color: #334155!important;
    font-weight: 700!important;
}
[data-testid="stMetricValue"] {
    color: #1e293b!important;
    font-size: 28px!important;
    font-weight: 700!important;
}
[data-testid="stMetricLabel"] {
    color: #475569!important;
    font-weight: 600!important;
}
[data-testid="stAlert"] {
    color: #1e293b!important;
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
.stNumberInput input {
    background: white!important;
    border: 1.5px solid #e2e8f0!important;
    border-radius: 12px!important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #8b5cf6!important;
    box-shadow: 0 0 3px rgba(139, 92, 246, 0.1)!important;
}
label {
    color: #475569!important;
    font-weight: 600!important;
}
/* SIMPLE CLEAN TITLE */
h1 {
    color: #0f172a!important;
    font-weight: 700!important;
    letter-spacing: -0.02em!important;
    font-size: 2rem!important;
    margin-bottom: 0.1rem!important;
}
[data-testid="stCaptionContainer"] {
    color: #64748b!important;
    font-size: 14px!important;
    font-weight: 500!important;
    margin-bottom: 1.5rem!important;
}
hr {
    margin: 1.5rem 0!important;
    border-color: #e2e8f0!important;
    border-width: 1px!important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #4f46e5 0%,
        #6366f1 100%
    );
} 
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6)!important;
    color: white!important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4)!important;
    border: none!important;
    font-weight: 600!important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5)!important;
    transform: translateY(-2px)!important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #14b8a6, #06b6d4)!important;
    box-shadow: 0 4px 14px rgba(20, 184, 166, 0.4)!important;
}
.stSlider {
    background: rgba(255,255,255,0.9);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
/* PROFESSIONAL MG LOGO */
.mg-logo {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    font-family: 'Arial', sans-serif;
    font-weight: 700;
    font-size: 20px;
    color: white;
    letter-spacing: -1px;
}
</style>
""", unsafe_allow_html=True)



# ---------------- LOAD MODEL ----------------
model = joblib.load("Artifacts/Models/ml_model.pkl")
feature_names = joblib.load("Artifacts/Models/feature_names.pkl")
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "report_data" not in st.session_state:
    st.session_state.report_data = {}

st.sidebar.markdown("""
# 🧠 AI Customer Intelligence

### Smart Customer Intelligence System

<div style="
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(12px);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.35);
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    margin-top: 15px;
">

<h4 style="
    color:white;
    text-align:center;
    margin-top:0;
    margin-bottom:15px;
    letter-spacing:0.5px;
">
✨ Platform Features
</h4>

<div style="
    color:white;
    line-height:2;
    font-size:14px;
">
🧠 Predict Customer Behavior<br>
📊 Interactive Analytics<br>
📈 Customer Trend Analysis<br>
⭐ Loyalty Insights<br>
📋 Smart Reports<br>
💾 Export Results
</div>

</div>

<div style="
    text-align:center;
    color:rgba(255,255,255,0.85);
    font-size:12px;
    margin-top:12px;
">
Powered by AI & Machine Learning
</div>
""", unsafe_allow_html=True)

menu = st.radio(
    "",
    ["🏠 Home", "🔮 Prediction", "📊 Analytics", "📋 Report", "📥 Export"],
    horizontal=True,
    key="main_menu"
)
st.sidebar.markdown("""
<div style="
width:70px;
height:70px;
border-radius:50%;
background:linear-gradient(135deg,#4f46e5,#7c3aed);
display:flex;
align-items:center;
justify-content:center;
margin:auto;
margin-bottom:20px;
">
<span style="
color:white;
font-size:26px;
font-weight:800;
">
MG
</span>
</div>
""", unsafe_allow_html=True)
# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Customer Intelligence",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>

/* ================= APP BACKGROUND ================= */
.stApp {
    background-color: #fafafa;
    background-image:
        radial-gradient(at 0% 0%, #fef3c7 0px, transparent 50%),
        radial-gradient(at 100% 0%, #dbeafe 0px, transparent 50%),
        radial-gradient(at 100% 100%, #e0e7ff 0px, transparent 50%),
        radial-gradient(at 0% 100%, #fce7f3 0px, transparent 50%);
    background-attachment: fixed;
    color: #0f172a;
}

/* ================= TEXT DEFAULT ================= */
h1 {
    color: #0f172a !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}

h2, h3, h4 {
    color: #334155 !important;
    font-weight: 700 !important;
}

label {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* ================= INPUTS ================= */
div[data-baseweb="select"] {
    background: white !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
}

div[data-baseweb="input"] > div,
.stNumberInput input {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #111827 !important;
}

/* ================= METRICS ================= */
[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-weight: 600 !important;
}

/* ================= BUTTONS ================= */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(99,102,241,0.35);
}

/* ================= SLIDER ================= */
.stSlider {
    background: #ffffff;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background: #eff6ff;
    border-right: 1px solid #dbeafe;
}

section[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

/* ================= TOP MENU (RADIO) ================= */
div[data-testid="stRadio"] label {
    color: #111827 !important;   /* DEFAULT BLACK */
    font-weight: 600 !important;
    padding: 8px 14px;
    border-radius: 10px;
    margin-right: 6px;
    transition: 0.2s ease;
}

/* HOVER */
div[data-testid="stRadio"] label:hover {
    background: #e5e7eb !important;
}
div[data-testid="stRadio"] label {
    color: #111827 !important;
    font-weight: 600 !important;
    padding: 8px 14px;
    border-radius: 10px;
}

/* when selected */
div[data-testid="stRadio"] input:checked + div {
    background: #4f46e5 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: 800 !important;
}           

/* ACTIVE (CLICKED TAB) */
div[data-testid="stRadio"] input:checked + div {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border-radius: 10px;
    font-weight: 800 !important;
    box-shadow: 0 4px 12px rgba(79,70,229,0.25);
}
div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* CONTAINER - SINGLE LINE */
div[data-testid="stRadio"] > div {
    flex-wrap: nowrap !important;
    gap: 20px !important;  /* TEXT KU NALLA GAP */
}

/* TEXT MATTUM */
div[data-testid="stRadio"] label p {
    color: #111827 !important;  /* DEFAULT BLACK */
    font-weight: 700 !important;
    font-size: 22px !important; /* PERUSU */
    cursor: pointer !important;
}

/* SELECTED TEXT COLOR */
div[data-testid="stRadio"] label[data-checked="true"] p {
    color: #4f46e5 !important;  /* PURPLE TEXT */
}

/* BOX/BORDER ELLAM REMOVE */
div[data-testid="stRadio"] label {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("Artifacts/Models/ml_model.pkl")
feature_names = joblib.load("Artifacts/Models/feature_names.pkl")
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "report_data" not in st.session_state:
    st.session_state.report_data = {}

if menu == "🏠 Home":
    # ---------------- PREMIUM HEADER ----------------
    col1, col2 = st.columns([8, 2])

    with col1:
        st.markdown("""
        <h1 style="
        font-size:3rem;
        font-weight:800;
        background:linear-gradient(90deg,#4f46e5,#7c3aed,#06b6d4);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        margin-bottom:0;">
        AI Customer Intelligence Platform
        </h1>

        <p style="
        color:#64748b;
        font-size:18px;
        font-weight:500;">
        Predict • Analyze • Optimize • Grow
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            width:100px;
            height:100px;
            border-radius:50%;
            background:linear-gradient(135deg,#4f46e5,#7c3aed);
            display:flex;
            align-items:center;
            justify-content:center;
            box-shadow:0 15px 35px rgba(79,70,229,0.4);
            margin-left:auto;
            border:4px solid white;
        ">
            <span style="
                color:white;
                font-size:34px;
                font-weight:900;
            ">
                MG
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    # ---------------- METRICS ----------------
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
       st.markdown("""
    <div style="
        padding:18px;
        height:130px;
        border-radius:18px;
        background:#EEF2FF;
        border-left:6px solid #6366F1;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
    ">
        <h4 style="margin:0;">👥  Customers</h4>
        <h2 style="margin:5px 0 0 0; font-size:28px; line-height:1;">3900</h2>
    </div>
    """, unsafe_allow_html=True)

    with col2:
     st.markdown("""
        <div style="
            padding:18px;
            height:130px;
            border-radius:18px;
            background:#ECFEFF;
            border-left:6px solid #06B6D4;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            text-align:center;
        ">
            <h4 style="margin:0;">📊 Features</h4>
            <h2 style="margin:5px 0 0 0; font-size:28px; line-height:1;">19</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
      st.markdown("""
        <div style="
            padding:18px;
            height:130px;
            border-radius:18px;
            background:#FEF3C7;
            border-left:6px solid #F59E0B;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            text-align:center;
        ">
            <h4 style="margin:0;">🤖 Model</h4>
            <h2 style="margin:5px 0 0 0; font-size:28px; line-height:1;">Logistic</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
     st.markdown("""
        <div style="
            padding:18px;
            height:130px;
            border-radius:18px;
            background:#ECFDF5;
            border-left:6px solid #10B981;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            text-align:center;
        ">
            <h4 style="margin:0;">✅ Status</h4>
            <h2 style="margin:5px 0 0 0; font-size:28px; line-height:1;">Ready</h2>
        </div>
        """, unsafe_allow_html=True)
        
    

    st.markdown("---")
    st.subheader("🚀 About Project")

    st.info("""
    AI Customer Intelligence Platform uses Machine Learning
    to predict customer behavior and identify high-value customers.
    The system provides analytics, recommendations, and reports
    to support business decision-making.
    """)    
elif menu == "🔮 Prediction":
    st.success("Prediction Page Loaded")
    # ---------------- INPUT ----------------
    st.markdown("""
    <div style='
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 32px rgba(15, 23, 42, 0.08);
    margin: 30px 0;
    '>
    <h3 style='color:#1e293b; margin-top:0;'>👤 Customer Information</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        rating = st.slider("Rating", 1.0, 5.0, 4.0)

    with col2:
        purchases = st.number_input("Previous Purchases", 0, 100, 10)
        category = st.selectbox("Category", ["Clothing", "Footwear", "Accessories", "Outerwear"])
        item = st.selectbox("Item", ["Shirt", "Pants", "Shoes", "Dress"])

    location = st.selectbox("Location", ["California", "Texas", "New York"])
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    payment = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Cash"])

    # ---------------- SCORES ----------------
    engagement = (rating * 20 + purchases) / 2
    risk = max(0, 100 - engagement)
    loyalty = min(100, engagement + rating * 10)
    customer_score = round(
    (engagement + loyalty - risk) / 2,
    2
)

    # ---------------- PREDICT ----------------
    if st.button("🔮 Predict Customer Behavior"):
        data = {
            'Age': age, 'Review Rating': rating, 'Previous Purchases': purchases,
            'Category_Avg_Purchase': 60, 'Location_Total_Revenue': 4500,
            'Item_Avg_Rating': 3.7, 'Payment_Method_Count': 650,
            f'Gender_{gender}': 1, f'Item Purchased_{item}': 1,
            f'Category_{category}': 1, f'Location_{location}': 1,
            f'Season_{season}': 1, f'Payment Method_{payment}': 1
        }

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        prediction = model.predict(input_df)
        try:
            prob = model.predict_proba(input_df)[0]
            confidence = max(prob)
        except:
            confidence = 0
        st.session_state["age"] = age
        st.session_state["purchases"] = purchases
        st.session_state["rating"] = rating
        st.session_state["engagement"] = engagement
        st.session_state["risk"] = risk
        st.success("✅ Data saved")   


        # ---------------- RESULT ----------------
        st.subheader("🎯 Prediction Result")
        if prediction[0] == 1:
                
                st.markdown("""
                <div style="
                background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
                border: 2px solid #10b981;
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 15px 50px rgba(16, 185, 129, 0.4);
                text-align: center;
                animation: glow 2s ease-in-out infinite;
                margin-bottom: 20px;
                ">
                <div style="font-size: 48px; margin-bottom: 10px;">🏆</div>
                <h2 style='margin:0; color:#065f46; font-size:32px; font-weight:800;'>VIP CUSTOMER UNLOCKED!</h2>
                <p style='margin:15px 0 0 0; color:#047857; font-size:18px; font-weight:500;'>
                High engagement detected • Premium conversion opportunity
                </p>
                </div>
                <style>
                @keyframes glow {
                    0%, 100% { box-shadow: 0 15px 50px rgba(16, 185, 129, 0.4); }
                    50% { box-shadow: 0 15px 70px rgba(16, 185, 129, 0.6); }
                }
                </style>
                """, unsafe_allow_html=True)

                status = "VIP CUSTOMER 🟢"
                recommendation = "🔥 Offer Premium Plans / Upsell Products"
        else:
            st.markdown("""
            <div style="
            background:linear-gradient(135deg,#fee2e2,#fecaca);
            padding:30px;
            border-radius:20px;
            text-align:center;
            border:2px solid #ef4444;">
            <h2>⚠️ CUSTOMER AT RISK</h2>
            <p>Retention campaign recommended</p>
            </div>
            """, unsafe_allow_html=True)

            status = "AT RISK 🔴"
            recommendation = "📢 Run Retention Campaign"
                # ---------------- CONFIDENCE SCORE ----------------
        st.subheader("📊 Confidence Score")

        st.markdown(f"""
        <div style="
        background:#EEF2FF;
        padding:15px;
        border-radius:12px;
        border-left:5px solid #6366F1;
        font-weight:600;
        font-size:18px;
        color:#1E293B;
        margin-bottom:10px;
        ">
        🎯 Confidence Score: {confidence*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(confidence))

        # ---------------- KEY METRICS ----------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 Confidence", f"{confidence*100:.1f}%")

        with col2:
            st.metric("💎 Loyalty", f"{loyalty:.1f}")

        with col3:
            st.metric("⚠️ Risk", f"{risk:.1f}")

        # ---------------- BUSINESS SCORES ----------------
        st.subheader("📊 Business Scores")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📈 Engagement", f"{engagement:.1f}")

        with col2:
            st.metric("💎 Loyalty", f"{loyalty:.1f}")

        with col3:
            st.metric("⚠️ Risk", f"{risk:.1f}")

        with col4:
            st.metric("⭐ Customer Score", customer_score)

        # ---------------- RECOMMENDATION ----------------
        st.subheader("💡 Recommendation System")
        st.success(recommendation)
        st.markdown(f"""
            <div style="
            background:white;
            padding:20px;
            border-radius:15px;
            border-left:6px solid #8b5cf6;
            box-shadow:0 4px 10px rgba(0,0,0,0.1);
            margin-top:10px;
            ">
            <h3>🤖 AI Insight</h3>

            <p>
            Customer Loyalty Score: {loyalty:.1f}<br>
            Risk Level: {risk:.1f}<br>
            Customer Score: {customer_score}<br>
            Recommended Action: {recommendation}
            </p>

            </div>
            """, unsafe_allow_html=True)

        # ---------------- CUSTOMER SUMMARY ----------------
        result = "Likely to Subscribe" if prediction[0] == 1 else "Unlikely"

        st.markdown(f"""
        <div style="
        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0 6px 20px rgba(0,0,0,0.08);
        border:1px solid #e2e8f0;
        margin:20px 0;
        ">
        <h3 style="color:#4f46e5;">👤 Customer Profile</h3>

        <p><b>Age:</b> {age}</p>
        <p><b>Gender:</b> {gender}</p>
        <p><b>Item:</b> {item}</p>
        <p><b>Category:</b> {category}</p>
        <p><b>Location:</b> {location}</p>
        <p><b>Season:</b> {season}</p>
        <p><b>Payment:</b> {payment}</p>
        <p><b>Prediction:</b> {result}</p>
        <p><b>Confidence:</b> {confidence*100:.2f}%</p>
        <p><b>Engagement:</b> {engagement:.1f}</p>
        <p><b>Loyalty:</b> {loyalty:.1f}</p>
        <p><b>Risk:</b> {risk:.1f}</p>
        <p><b>Recommendation:</b> {recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
elif menu == "📊 Analytics":

    st.title("📊 Analytics Dashboard")
    st.subheader("📈 Customer Analytics")
    st.info("Visual analysis based on latest prediction.")

    if "age" not in st.session_state:
        st.warning("⚠️ Run a prediction first.")

    else:

        st.success("✅ Analytics Loaded")

        # BAR CHART
        fig, ax = plt.subplots(figsize=(5,5))

        ax.bar(
            ["Age", "Purchases", "Rating"],
            [
                st.session_state["age"],
                st.session_state["purchases"],
                st.session_state["rating"] * 20
            ],
            color=[
        "#6F71EC",  # Blue - Age
        "#14B8A6",  # Teal - Purchases
        "#F59E0B"   # Orange - Rating
    ]
)

        ax.set_title("Customer Behavior Analysis")

        st.pyplot(fig)

        # RADAR CHART
        labels = ["Age", "Engagement", "Rating", "Purchases", "Risk"]

        values = [
            st.session_state["age"] / 100,
            st.session_state["engagement"] / 100,
            st.session_state["rating"] / 5,
            st.session_state["purchases"] / 100,
            st.session_state["risk"] / 100
        ]

        angles = np.linspace(
            0,
            2 * np.pi,
            len(labels),
            endpoint=False
        ).tolist()

        values += values[:1]
        angles += angles[:1]

        fig2 = plt.figure(figsize=(5,5))
        ax2 = plt.subplot(111, polar=True)

        ax2.plot(angles, values)
        ax2.fill(angles, values, alpha=0.25)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels)

        st.pyplot(fig2)
        
elif menu == "🔮 Prediction":
    st.success("Prediction Page Loaded")
    # ---------------- INPUT ----------------
    st.markdown("""
    <div style='
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 32px rgba(15, 23, 42, 0.08);
    margin: 30px 0;
    '>
    <h3 style='color:#1e293b; margin-top:0;'>👤 Customer Information</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        rating = st.slider("Rating", 1.0, 5.0, 4.0)

    with col2:
        purchases = st.number_input("Previous Purchases", 0, 100, 10)
        category = st.selectbox("Category", ["Clothing", "Footwear", "Accessories", "Outerwear"])
        item = st.selectbox("Item", ["Shirt", "Pants", "Shoes", "Dress"])

    location = st.selectbox("Location", ["California", "Texas", "New York"])
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    payment = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Cash"])

    # ---------------- SCORES ----------------
    engagement = (rating * 20 + purchases) / 2
    risk = max(0, 100 - engagement)
    loyalty = min(100, engagement + rating * 10)
    customer_score = round(
    (engagement + loyalty - risk) / 2,
    2
)

    # ---------------- PREDICT ----------------
    if st.button("🔮 Predict Customer Behavior"):
        data = {
            'Age': age, 'Review Rating': rating, 'Previous Purchases': purchases,
            'Category_Avg_Purchase': 60, 'Location_Total_Revenue': 4500,
            'Item_Avg_Rating': 3.7, 'Payment_Method_Count': 650,
            f'Gender_{gender}': 1, f'Item Purchased_{item}': 1,
            f'Category_{category}': 1, f'Location_{location}': 1,
            f'Season_{season}': 1, f'Payment Method_{payment}': 1
        }

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        prediction = model.predict(input_df)
        try:
            prob = model.predict_proba(input_df)[0]
            confidence = max(prob)
        except:
            confidence = 0
        st.session_state["age"] = age
        st.session_state["purchases"] = purchases
        st.session_state["rating"] = rating
        st.session_state["engagement"] = engagement
        st.session_state["risk"] = risk
        # ---------------- RESULT ----------------
        st.subheader("🎯 Prediction Result")
        if prediction[0] == 1:
                
                st.markdown("""
                <div style="
                background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
                border: 2px solid #10b981;
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 15px 50px rgba(16, 185, 129, 0.4);
                text-align: center;
                animation: glow 2s ease-in-out infinite;
                margin-bottom: 20px;
                ">
                <div style="font-size: 48px; margin-bottom: 10px;">🏆</div>
                <h2 style='margin:0; color:#065f46; font-size:32px; font-weight:800;'>VIP CUSTOMER UNLOCKED!</h2>
                <p style='margin:15px 0 0 0; color:#047857; font-size:18px; font-weight:500;'>
                High engagement detected • Premium conversion opportunity
                </p>
                </div>
                <style>
                @keyframes glow {
                    0%, 100% { box-shadow: 0 15px 50px rgba(16, 185, 129, 0.4); }
                    50% { box-shadow: 0 15px 70px rgba(16, 185, 129, 0.6); }
                }
                </style>
                """, unsafe_allow_html=True)

                status = "VIP CUSTOMER 🟢"
                recommendation = "🔥 Offer Premium Plans / Upsell Products"
        else:
            st.markdown("""
            <div style="
            background:linear-gradient(135deg,#fee2e2,#fecaca);
            padding:30px;
            border-radius:20px;
            text-align:center;
            border:2px solid #ef4444;">
            <h2>⚠️ CUSTOMER AT RISK</h2>
            <p>Retention campaign recommended</p>
            </div>
            """, unsafe_allow_html=True)

            status = "AT RISK 🔴"
            recommendation = "📢 Run Retention Campaign"
                # ---------------- CONFIDENCE SCORE ----------------
        st.subheader("📊 Confidence Score")

        st.markdown(f"""
        <div style="
        background:#EEF2FF;
        padding:15px;
        border-radius:12px;
        border-left:5px solid #6366F1;
        font-weight:600;
        font-size:18px;
        color:#1E293B;
        margin-bottom:10px;
        ">
        🎯 Confidence Score: {confidence*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(confidence))

        # ---------------- KEY METRICS ----------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 Confidence", f"{confidence*100:.1f}%")

        with col2:
            st.metric("💎 Loyalty", f"{loyalty:.1f}")

        with col3:
            st.metric("⚠️ Risk", f"{risk:.1f}")

        # ---------------- BUSINESS SCORES ----------------
        st.subheader("📊 Business Scores")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
         st.metric("🎯 Confidence", f"{confidence*100:.1f}%")

        with col2:
         st.metric("💎 Loyalty", f"{loyalty:.1f}")

        with col3:
         st.metric("⚠️ Risk", f"{risk:.1f}")

        with col4:
         st.metric("⭐ Customer Score", customer_score)

        # ---------------- RECOMMENDATION ----------------
        st.subheader("💡 Recommendation System")
        st.success(recommendation)
        st.markdown(f"""
        <div style="
        background:white;
        padding:20px;
        border-radius:15px;
        border-left:6px solid #8b5cf6;
        box-shadow:0 4px 10px rgba(0,0,0,0.1);
        margin-top:10px;
        ">
        <h3>🤖 AI Insight</h3>

        <p>
        Customer Loyalty Score: {loyalty:.1f}<br>
        Risk Level: {risk:.1f}<br>
        Customer Score: {customer_score}<br>
        Recommended Action: {recommendation}
        </p>

        </div>
        """, unsafe_allow_html=True)

        # ---------------- CUSTOMER SUMMARY ----------------
        result = "Likely to Subscribe" if prediction[0] == 1 else "Unlikely"

        st.markdown(f"""
        <div style="
        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0 6px 20px rgba(0,0,0,0.08);
        border:1px solid #e2e8f0;
        margin:20px 0;
        ">
        <h3 style="color:#4f46e5;">👤 Customer Profile</h3>

        <p><b>Age:</b> {age}</p>
        <p><b>Gender:</b> {gender}</p>
        <p><b>Item:</b> {item}</p>
        <p><b>Category:</b> {category}</p>
        <p><b>Location:</b> {location}</p>
        <p><b>Season:</b> {season}</p>
        <p><b>Payment:</b> {payment}</p>
        <p><b>Prediction:</b> {result}</p>
        <p><b>Confidence:</b> {confidence*100:.2f}%</p>
        <p><b>Engagement:</b> {engagement:.1f}</p>
        <p><b>Loyalty:</b> {loyalty:.1f}</p>
        <p><b>Risk:</b> {risk:.1f}</p>
        <p><b>Recommendation:</b> {recommendation}</p>
        </div>
        """, unsafe_allow_html=True)

    st.write(st.session_state)

    if "age" not in st.session_state:
        st.warning("⚠️ Run a prediction first.")
elif menu == "📋 Report":

    st.title("📋 Customer Report")

    if "age" not in st.session_state:
        st.warning("⚠️ Run a prediction first.")

    else:

        st.success("✅ Report Generated")

        st.markdown(f"""
        <div style="
        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0 6px 20px rgba(0,0,0,0.08);
        border:1px solid #e2e8f0;
        ">

        <h2 style="color:#4f46e5;">
        👤 Customer Summary Report
        </h2>

        <hr>

        <p><b>Age:</b> {st.session_state["age"]}</p>

        <p><b>Previous Purchases:</b>
        {st.session_state["purchases"]}</p>

        <p><b>Review Rating:</b>
        {st.session_state["rating"]}</p>

        <p><b>Engagement Score:</b>
        {st.session_state["engagement"]:.1f}</p>

        <p><b>Risk Score:</b>
        {st.session_state["risk"]:.1f}</p>

        </div>
        """, unsafe_allow_html=True)
        st.subheader("📈 Report Insights")

    if  st.session_state["risk"] > 50:
        st.error("High Risk Customer - Retention campaign recommended.")
    
    elif st.session_state["risk"] > 30:  # Medium risk ku yellow box
        st.markdown("""
        <div style="
            background-color: #fef9c3; 
            color: #713f12; 
            padding: 12px 16px; 
            border-radius: 8px;
            border-left: 4px solid #eab308;
            font-weight: 600;
            font-size: 16px;
            margin-top: 10px;
        ">
            ⚡ Medium Risk - Customer engagement is low, target with email campaigns
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.success("Healthy Customer - Upsell opportunities available.")

        

        if st.session_state["engagement"] > 50:
            st.info(
                "Customer engagement is above average."
            )
        else:
            st.warning(
                "Customer engagement is below average."
            )

elif menu == "📥 Export":

    st.title("📥 Export Report")

    if "age" not in st.session_state:
        st.warning("⚠️ Run a prediction first.")

    else:

        report = pd.DataFrame({
            "Age": [st.session_state["age"]],
            "Purchases": [st.session_state["purchases"]],
            "Rating": [st.session_state["rating"]],
            "Engagement": [round(st.session_state["engagement"], 2)],
            "Risk": [round(st.session_state["risk"], 2)]
        })

        st.dataframe(report, use_container_width=True)

        st.download_button(
            label="📥 Download CSV Report",
            data=report.to_csv(index=False),
            file_name="customer_report.csv",
            mime="text/csv"
        )

        st.success("✅ Report ready for download")