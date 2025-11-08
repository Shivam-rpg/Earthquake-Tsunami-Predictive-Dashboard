import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="🌊 Tsunami Risk Prediction Dashboard",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM STYLING ======
st.markdown("""
<style>
/* Title */
h1 {
    color: #003366;
    text-align: center;
    font-weight: 900 !important;
}
div[data-testid="stMetricValue"] {
    color: #0d6efd;
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f4f7fb;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    return joblib.load("Tsunami_model.pkl")

model = load_model()

# ====== HEADER ======
st.title("🌋 Earthquake & Tsunami Risk Prediction Dashboard")
st.markdown("""
Welcome to the **AI-powered Tsunami Risk Prediction System**,  
where machine learning meets real-time seismic insights 🌍.

Use the panel on the left to input earthquake details and assess tsunami likelihood.
""")

st.markdown("---")

# ====== SIDEBAR INPUT ======
st.sidebar.header("⚙️ Input Earthquake Parameters")

magnitude = st.sidebar.slider("Magnitude (Mw)", 4.0, 10.0, 6.5)
depth = st.sidebar.slider("Depth (km)", 0.0, 700.0, 50.0)
cdi = st.sidebar.slider("CDI (Community Intensity)", 0, 10, 5)
mmi = st.sidebar.slider("MMI (Modified Mercalli)", 0, 10, 5)
sig = st.sidebar.number_input("Significance (sig)", 0, 1000, 600)
nst = st.sidebar.number_input("Number of Stations (nst)", 0, 1000, 100)
dmin = st.sidebar.number_input("Dmin (Distance to nearest station)", 0.0, 10.0, 1.0)
gap = st.sidebar.slider("Gap (degrees)", 0.0, 180.0, 40.0)
latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, 0.0)
longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, 0.0)
year = st.sidebar.slider("Year", 1900, 2025, 2024)
month = st.sidebar.slider("Month", 1, 12, datetime.now().month)

# ====== INPUT DATA ======
input_data = pd.DataFrame([{
    "magnitude": magnitude, "cdi": cdi, "mmi": mmi, "sig": sig,
    "nst": nst, "dmin": dmin, "gap": gap, "depth": depth,
    "latitude": latitude, "longitude": longitude, "Year": year, "Month": month
}])

st.write("### 📋 Input Summary")
st.dataframe(input_data.style.highlight_max(axis=0), use_container_width=True)

# ====== PREDICTION BUTTON ======
if st.button("🚀 Run Tsunami Risk Prediction", use_container_width=True):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    # ====== PREDICTION OUTPUT ======
    with col1:
        if prediction == 1:
            st.markdown("### 🌊 **High Tsunami Probability Detected**")
            st.warning("⚠️ **Immediate caution advised!**\nThe model detects a significant chance of tsunami occurrence.")
        else:
            st.markdown("### ✅ **No Major Tsunami Risk**")
            st.success("Safe conditions predicted — no significant tsunami risk detected.")
        st.metric("Tsunami Risk Probability", f"{prob:.2f} %")

    # ====== GAUGE ======
    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob,
            title={'text': "Risk Level (%)", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "#007BFF"},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "gold"},
                    {'range': [70, 100], 'color': "tomato"}
                ],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ====== TIMESTAMP ======
    with col3:
        st.info(f"🕒 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown("---")

    # ====== MAP ======
    st.subheader("🌍 Global Earthquake Visualization")

    map_df = pd.DataFrame({
        "Latitude": [latitude],
        "Longitude": [longitude],
        "Magnitude": [magnitude],
        "Depth (km)": [depth],
        "Risk Level": ["High" if magnitude >= 7 else "Moderate" if magnitude >= 5 else "Low"]
    })

    fig_map = px.scatter_mapbox(
        map_df,
        lat="Latitude",
        lon="Longitude",
        color="Risk Level",
        size="Magnitude",
        hover_name="Risk Level",
        hover_data={"Magnitude": True, "Depth (km)": True},
        color_discrete_map={"Low": "green", "Moderate": "orange", "High": "red"},
        size_max=30,
        zoom=2,
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=2.5,
        mapbox_center={"lat": latitude, "lon": longitude},
        margin={"r":0, "t":0, "l":0, "b":0},
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # ====== FEATURE IMPORTANCE ======
    st.subheader("📊 Model Feature Importance")

    try:
        feat_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Feature Influence on Tsunami Prediction"
        )

        fig_imp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13)
        )

        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Feature importance is not available for this model type.")
else:
    st.info("👈 Adjust earthquake parameters and click **Run Tsunami Risk Prediction** to begin.")








