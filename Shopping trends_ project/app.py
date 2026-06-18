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
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #334155 100%);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(90deg,#6366f1,#22c55e);
    color: white;
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
with col1:
    st.metric("👥 Customers", "3900")
with col2:
    st.metric("📊 Features", "19")
with col3:
    st.metric("🤖 Model", "Logistic")
with col4:
    st.metric("✅ Status", "Ready")

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

    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="white")

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

    ax2.plot(angles, values, color="cyan", linewidth=2)
    ax2.fill(angles, values, color="cyan", alpha=0.2)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, color="white", fontsize=9)

    fig2.patch.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

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