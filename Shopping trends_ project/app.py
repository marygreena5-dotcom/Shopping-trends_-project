import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
import time

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
    background: linear-gradient(
        135deg,
        #fdfbff 0%,
        #eef2ff 35%,
        #e0f2fe 70%,
        #f0fdf4 100%
    );
    color: #1e293b;
}
h1, h2, h3, p, label {
    color: #1e293b !important;
}
/* Input Boxes */
div[data-baseweb="input"] > div {
    background: white !important;
    border: 1px solid #dbeafe !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.08);
}

div[data-baseweb="input"] input {
    color: #1e293b !important;
}

/* Dropdowns */
div[data-baseweb="select"] > div {
    background: white !important;
    border: 1px solid #dbeafe !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.08);
}

/* Labels */
label {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* Slider */
.stSlider {
    background: rgba(255,255,255,0.8);
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #dbeafe;
    box-shadow: 0 2px 8px rgba(59,130,246,0.08);
}

/* Number Input */
[data-testid="stNumberInput"] {
    background: white;
    padding: 8px;
    border-radius: 12px;
}           

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
}

/* Predict Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg,#4f46e5,#7c3aed);
    color: white;
    box-shadow: 0 6px 18px rgba(124,58,237,0.25);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(124,58,237,0.35);
}

/* Download Button */
.stDownloadButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color: white;
    box-shadow: 0 6px 18px rgba(59,130,246,0.25);
    transition: all 0.3s ease;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(59,130,246,0.35);
}
</style>
""", unsafe_allow_html=True)

# ---------------- GLITTER EFFECT ----------------
def show_glitter():
    glitter_html = ""
    for _ in range(30):
        left = random.randint(0, 100)
        delay = random.uniform(0, 2)
        size = random.randint(3, 7)

        glitter_html += f"""
        <div style="
            position:fixed;
            top:-10px;
            left:{left}%;
            width:{size}px;
            height:{size}px;
            background:white;
            border-radius:50%;
            animation: fall 3s linear forwards;
            animation-delay:{delay}s;
            z-index:9999;
            opacity:0.8;
        "></div>
        """

    st.markdown(glitter_html, unsafe_allow_html=True)

    st.markdown("""
    <style>
    @keyframes fall {
        0% {transform: translateY(0); opacity: 1;}
        100% {transform: translateY(100vh); opacity: 0;}
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("Artifacts/Models/ml_model.pkl")
feature_names = joblib.load("Artifacts/Models/feature_names.pkl")

# ---------------- HEADER ----------------
st.title("🛍️ Smart Customer Subscription Predictor")
st.markdown("### AI Dashboard for Customer Behavior & Insights")
# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)
card_style = """
padding:22px;
border-radius:20px;
background:rgba(255,255,255,0.85);
border:1px solid rgba(255,255,255,0.6);
backdrop-filter:blur(10px);
box-shadow:0 10px 25px rgba(15,23,42,0.08);
color:#1e293b;
text-align:center;
"""

with col1:
    st.markdown(f"""
    <div style="{card_style} border-top:4px solid #60a5fa;">
        <h4>👥 Customers</h4>
        <h2>3900</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="{card_style} border-top:4px solid #818cf8;">
        <h4>📊 Features</h4>
        <h2>19</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="{card_style} border-top:4px solid #2dd4bf;">
        <h4>🤖 Model</h4>
        <h2>Logistic</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="{card_style} border-top:4px solid #4ade80;">
        <h4>✅ Status</h4>
        <h2>Ready</h2>
    </div>
    """, unsafe_allow_html=True)

# ---------------- INPUT ----------------
st.subheader("👤 Customer Information")

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

# ---------------- PREDICT ----------------
if st.button("🔮 Predict Customer Behavior"):

    data = {
        'Age': age,
        'Review Rating': rating,
        'Previous Purchases': purchases,
        'Category_Avg_Purchase': 60,
        'Location_Total_Revenue': 4500,
        'Item_Avg_Rating': 3.7,
        'Payment_Method_Count': 650,
        f'Gender_{gender}': 1,
        f'Item Purchased_{item}': 1,
        f'Category_{category}': 1,
        f'Location_{location}': 1,
        f'Season_{season}': 1,
        f'Payment Method_{payment}': 1
    }

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(input_df)

    try:
        prob = model.predict_proba(input_df)[0]
        confidence = max(prob)
        has_prob = True
    except:
        confidence = 0
        has_prob = False

    # ---------------- RESULT ----------------
    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.success("🎉 HIGH VALUE CUSTOMER")
        show_glitter()

        st.markdown("""
        <div style="
        padding:25px;
        border-radius:15px;
        background:linear-gradient(90deg,#16a34a,#22c55e);
        text-align:center;
        font-size:22px;
        font-weight:bold;
        color:white;
        box-shadow:0px 0px 25px rgba(34,197,94,0.7);
        ">
        🚀 VIP CUSTOMER DETECTED
        </div>
        """, unsafe_allow_html=True)

        status = "VIP CUSTOMER 🟢"
        recommendation = "🔥 Offer Premium Plans / Upsell Products"

    else:
        st.error("❌ LOW ENGAGEMENT")
        st.snow()
        status = "AT RISK 🔴"
        recommendation = "📢 Run Marketing Campaign / Give Discounts"

    # ---------------- CONFIDENCE ----------------
    st.subheader("📊 Confidence Score")
    st.progress(float(confidence))
    st.info(f"Confidence: {confidence*100:.2f}%")

    # ---------------- ENGAGEMENT + LOYALTY + RISK ----------------
    st.subheader("📊 Business Scores")

    st.write(f"Engagement Score: **{engagement:.2f}**")
    st.write(f"Loyalty Score: **{loyalty:.2f}**")
    st.write(f"Risk Score: **{risk:.2f}**")

    # ---------------- RECOMMENDATION ----------------
    st.subheader("💡 Recommendation System")
    st.success(recommendation)

    # ---------------- SUMMARY ----------------
    st.subheader("📋 Customer Summary")

    summary_df = pd.DataFrame({
        "Feature": ["Age", "Gender", "Item", "Category", "Location", "Season", "Payment"],
        "Value": [age, gender, item, category, location, season, payment]
    })

    st.dataframe(summary_df, use_container_width=True)

    # ---------------- ANALYTICS ----------------
    st.subheader("📊 Behavior Chart")

    fig, ax = plt.subplots()

    ax.bar(
        ["Age", "Purchases", "Rating"],
        [age, purchases, rating * 20],
        color=["#3b82f6", "#22c55e", "#f97316"]
    )

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#334155")

    st.pyplot(fig)

    # ---------------- RADAR (SMALL FIXED SIZE) ----------------
    st.subheader("📡 Radar View")

    labels = ["Age", "Engagement", "Rating", "Purchases", "Risk"]
    values = [age/100, engagement, rating*20, purchases, risk]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig2 = plt.figure(figsize=(3.5, 3.5))
    ax2 = plt.subplot(111, polar=True)

    # Purple theme matching button
    ax2.plot(angles, values, color="#7c3aed", linewidth=2)
    ax2.fill(angles, values, color="#7c3aed", alpha=0.2)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, color="#334155", fontsize=9)

    # Light theme background
    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    # Grid color
    ax2.grid(color="#cbd5e1")

    st.pyplot(fig2)

    # ---------------- DOWNLOAD ----------------
    st.subheader("📥 Download Report")

    result = "Likely to Subscribe" if prediction[0] == 1 else "Unlikely"

    report = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Item": [item],
        "Category": [category],
        "Location": [location],
        "Prediction": [result],
        "Status": [status],
        "Confidence": [confidence],
        "Engagement": [engagement],
        "Loyalty": [loyalty],
        "Risk": [risk]
    })

    st.download_button(
        "Download Report",
        report.to_csv(index=False),
        file_name="customer_report.csv",
        mime="text/csv"
    )
st.markdown("---")

st.markdown(
     """
    <div style='text-align:center; color:#64748b; padding:10px;'>
    Developed by Greena | AI Customer Intelligence Dashboard 🚀
    </div>
    """,
    unsafe_allow_html=True
    )