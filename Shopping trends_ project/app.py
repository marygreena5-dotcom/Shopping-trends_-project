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
    padding:12px;
    border-radius:18px;
    background:#EEF2FF;
    border-left:6px solid #6366F1;
    text-align:center;">
    <h4>👥 Customers</h4>
    <h2>3900</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
    padding:12px;
    border-radius:18px;
    background:#ECFEFF;
    border-left:6px solid #06B6D4;
    text-align:center;">
    <h4>📊 Features</h4>
    <h2>19</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
    padding:12px;
    border-radius:18px;
    background:#FEF3C7;
    border-left:6px solid #F59E0B;
    text-align:center;">
    <h4>🤖 Model</h4>
    <h2>Logistic</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="
    padding:12px;
    border-radius:18px;
    background:#ECFDF5;
    border-left:6px solid #10B981;
    text-align:center;">
    <h4>✅ Status</h4>
    <h2>Ready</h2>
    </div>
    """, unsafe_allow_html=True)

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
        st.error("❌ LOW ENGAGEMENT")
        status = "AT RISK 🔴"
        recommendation = "📢 Run Marketing Campaign / Give Discounts"

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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Engagement", f"{engagement:.1f}")
    with col2:
        st.metric("Loyalty", f"{loyalty:.1f}")
    with col3:
        st.metric("Risk", f"{risk:.1f}")

    # ---------------- RECOMMENDATION ----------------
    st.subheader("💡 Recommendation System")
    st.success(recommendation)

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
    <table style="width:100%; border-collapse:collapse;">
    <tr style="background:#f8fafc;">
    <td style="padding:12px;"><b>Age</b></td>
    <td style="padding:12px;">{age}</td>
    </tr>
    <tr>
    <td style="padding:12px;"><b>Gender</b></td>
    <td style="padding:12px;">{gender}</td>
    </tr>
    <tr style="background:#f8fafc;">
    <td style="padding:12px;"><b>Item Purchased</b></td>
    <td style="padding:12px;">{item}</td>
    </tr>
    <td style="padding:12px;"><b>Category</b></td>
    <td style="padding:12px;">{category}</td>
    </tr>
    <tr style="background:#f8fafc;">
    <td style="padding:12px;"><b>Location</b></td>
    <td style="padding:12px;">{location}</td>
    </tr>
    <tr>
    <td style="padding:12px;"><b>Season</b></td>
    <td style="padding:12px;">{season}</td>
    </tr>
    <tr style="background:#f8fafc;">
    <td style="padding:12px;"><b>Payment Method</b></td>
    <td style="padding:12px;">{payment}</td>
    </tr>
    <tr>
    <td style="padding:12px;"><b>Prediction</b></td>
    <td style="padding:12px;"><b>{result}</b></td>
    </tr>
    <tr style="background:#f8fafc;">
    <td style="padding:12px;"><b>Confidence</b></td>
    <td style="padding:12px;">{confidence*100:.2f}%</td>
    </tr>
    <tr>
    <td style="padding:12px;"><b>Engagement Score</b></td>
    <td style="padding:12px;">{engagement:.1f}</td>
    </tr>
    <tr style="background:#f8fafc;">
    <td style="padding:12px;"><b>Loyalty Score</b></td>
    <td style="padding:12px;">{loyalty:.1f}</td>
    </tr>
    <tr>
    <td style="padding:12px;"><b>Risk Score</b></td>
    <td style="padding:12px;">{risk:.1f}</td>
    </tr>
    <tr style="background:#eef2ff;">
    <td style="padding:12px;"><b>Recommendation</b></td>
    <td style="padding:12px;">{recommendation}</td>
    </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- BEHAVIOR CHART ----------------
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(
        ["Age", "Purchases", "Rating"],
        [age, purchases, rating * 20]
    )
    colors = ["#6366f1", "#8b5cf6", "#14b8a6"]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_edgecolor("white")
        bar.set_linewidth(2)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title(
    "Customer Behavior Analysis",
    fontsize=14,
    fontweight="bold"
)

    # ---------------- RADAR CHART ----------------
    labels = ["Age", "Engagement", "Rating", "Purchases", "Risk"]
    values = [age/100, engagement/100, rating/5, purchases/100, risk/100]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig2 = plt.figure(figsize=(5, 5))
    ax2 = plt.subplot(111, polar=True)
    ax2.plot(angles, values, color="#6366f1", linewidth=3)
    ax2.fill(angles, values, color="#6366f1", alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_yticklabels([])
    ax2.grid(linestyle="--", alpha=0.5)
    ax2.set_title(
    "Customer Profile Radar",
    fontsize=14,
    fontweight="bold",
    pad=20
)

    # ---------------- CONFIDENCE CHART ----------------
    fig3, ax3 = plt.subplots(figsize=(6, 2))
    ax3.barh(["Confidence"], [confidence * 100], color="#14b8a6")
    ax3.set_xlim(0, 100)
    ax3.text(confidence * 100, 0, f"{confidence*100:.1f}%", va="center", ha="left", fontsize=12, fontweight="bold")
    ax3.set_facecolor("#f8fafc")
    fig3.patch.set_facecolor("#ffffff")
    ax3.set_title(
    "Prediction Confidence Score",
    fontsize=14,
    fontweight="bold"
)

    # ---------------- DISPLAY CHARTS ----------------
    st.subheader("📊 Customer Insights")
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Behavior Breakdown**")
        st.pyplot(fig)
    with col2:
        st.markdown("**🎯 Profile Radar**")
        st.pyplot(fig2)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("**💯 Prediction Confidence**")
    st.pyplot(fig3)

    # ---------------- DOWNLOAD ----------------
    st.subheader("📥 Download Report")
    report = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Item": [item],
    "Category": [category],
    "Location": [location],
    "Prediction": [result],
    "Status": [status],
    "Confidence": [round(confidence * 100, 2)],
    "Engagement": [round(engagement, 2)],
    "Loyalty": [round(loyalty, 2)],
    "Risk": [round(risk, 2)],
    "Recommendation": [recommendation]
    })
    st.download_button("Download Report", report.to_csv(index=False), file_name="customer_report.csv", mime="text/csv")

st.markdown("---")
st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:25px;
background:white;
border-radius:20px;
box-shadow:0 4px 20px rgba(0,0,0,.05);
margin-top:30px;
">

<h3 style="color:#4f46e5;">
🚀 AI Customer Intelligence Dashboard
</h3>

<p style="color:#64748b;">
Developed by <b>Greena</b> • Powered by Machine Learning & Streamlit
</p>

<p style="color:#94a3b8;">
Predict • Analyze • Optimize • Grow
</p>

</div>
""", unsafe_allow_html=True)