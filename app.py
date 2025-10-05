# app.py - BI-style AI Sales Dashboard with theme toggle & tracking
import traceback
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from prophet import Prophet

# ===== Sidebar: Theme Toggle =====
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])

# ===== Colors Based on Theme =====
if theme == "Dark":
    bg_color = "#0c1c3b"
    sidebar_color = "#111b2c"
    text_color = "#ffffff"
    table_bg1 = "#1a2a4c"
    table_bg2 = "#16203f"
    line_colors = ["#00ffcc","#ff9900","#00ccff","#ff6600"]
else:
    bg_color = "#f5f5f5"
    sidebar_color = "#e0e0e0"
    text_color = "#0c1c3b"
    table_bg1 = "#ffffff"
    table_bg2 = "#e8e8e8"
    line_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]

# ===== Apply CSS =====
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg_color};
    color: {text_color};
    font-family: "Arial", sans-serif;
}}
[data-testid="stSidebar"] {{
    background-color: {sidebar_color};
    color: {text_color};
}}
.stDataFrame tbody tr:nth-child(odd) {{ background-color: {table_bg1} !important; }}
.stDataFrame tbody tr:nth-child(even) {{ background-color: {table_bg2} !important; }}
.stDataFrame thead tr th {{ background-color: {sidebar_color} !important; color: {text_color}; }}
h1, h2, h3, h4, h5, h6, p {{ color: {text_color}; }}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AI Sales BI Dashboard", layout="wide")

# ===== Helpers =====
def make_sample_data(days=180):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    vals = (100 + pd.Series(range(days))*0.2 + pd.Series(range(days)).rolling(7).mean().fillna(0)).values
    return pd.DataFrame({"ds": dates, "y": vals})

@st.cache_data
def get_data(path="data/sales.csv"):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(p, parse_dates=[0])
    if "date" in df.columns: df = df.rename(columns={"date":"ds"})
    if "sales" in df.columns: df = df.rename(columns={"sales":"y"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df[["ds","y"]].sort_values("ds").reset_index(drop=True)

# ===== Sidebar: Inputs =====
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
model_choice = st.sidebar.selectbox("Model", ["Prophet"])
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, 28)
st.sidebar.markdown("---")
st.sidebar.write("Use CSV with 'date' and 'sales' columns. If not uploaded, sample data will be used.")

# ===== Load Data =====
df = None
if uploaded:
    try:
        df = pd.read_csv(uploaded, parse_dates=["date","ds"])
        if "date" in df.columns: df = df.rename(columns={"date":"ds"})
        if "sales" in df.columns: df = df.rename(columns={"sales":"y"})
        df = df[["ds","y"]].sort_values("ds").reset_index(drop=True)
    except:
        st.sidebar.error("CSV parse error. Using sample data.")
        df = make_sample_data()
else:
    try: df = get_data("data/sales.csv")
    except: df = make_sample_data()

# ===== KPI Tracking =====
df["daily_change"] = df["y"].diff()
df["weekly_sum"] = df["y"].rolling(7).sum()
df["cum_sales"] = df["y"].cumsum()

total_sales = df['y'].sum()
avg_sales = df['y'].mean()
last_sale = df['y'].iloc[-1]
max_sale = df['y'].max()
min_sale = df['y'].min()
daily_change = df['daily_change'].iloc[-1]
weekly_growth = df['weekly_sum'].iloc[-1] - df['weekly_sum'].iloc[-8] if len(df)>=8 else 0

st.markdown("<h1>AI Sales BI Dashboard</h1>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)
kpi1.metric("Total Sales", f"{total_sales:,.0f}")
kpi2.metric("Avg Daily Sales", f"{avg_sales:.2f}")
kpi3.metric("Latest Sale", f"{last_sale:.0f}")
kpi4.metric("Max Sale", f"{max_sale:.0f}")
kpi5.metric("Min Sale", f"{min_sale:.0f}")
kpi6.metric("Daily Change", f"{daily_change:+.0f}")
kpi7.metric("Weekly Growth", f"{weekly_growth:+.0f}")

st.markdown("---")
st.subheader("Recent Sales Data")
st.dataframe(df.tail(10))

# ===== Forecast =====
if st.button("Run Forecast"):
    with st.spinner("Forecasting..."):
        try:
            if Prophet is None: st.error("Prophet not installed.")
            else:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                m.fit(df[["ds","y"]])
                future = m.make_future_dataframe(periods=horizon)
                fcst = m.predict(future)

                # Forecast Line
                fig1 = px.line(fcst, x='ds', y='yhat', title=f"Forecast ({horizon} Days)",
                               labels={"yhat":"Forecast","ds":"Date"}, template="plotly_dark" if theme=="Dark" else "plotly_white",
                               color_discrete_sequence=[line_colors[0]])
                st.plotly_chart(fig1, use_container_width=True)

                # Actual vs Forecast
                merged = fcst.merge(df, on="ds", how="left")
                fig2 = px.line(merged, x='ds', y=['y','yhat'],
                               labels={"value":"Sales","variable":"Series"},
                               template="plotly_dark" if theme=="Dark" else "plotly_white",
                               color_discrete_sequence=line_colors[:2])
                st.plotly_chart(fig2, use_container_width=True)

                # Sales Distribution
                fig3 = px.histogram(df, x='y', nbins=20, title="Sales Distribution",
                                    template="plotly_dark" if theme=="Dark" else "plotly_white",
                                    color_discrete_sequence=[line_colors[3]])
                st.plotly_chart(fig3, use_container_width=True)

                # Cumulative Sales
                fig4 = px.area(df, x='ds', y='cum_sales', title="Cumulative Sales",
                               template="plotly_dark" if theme=="Dark" else "plotly_white",
                               color_discrete_sequence=[line_colors[2]])
                st.plotly_chart(fig4, use_container_width=True)

                # Sales by Weekday Pie
                df['weekday'] = df['ds'].dt.day_name()
                weekday_sum = df.groupby('weekday')['y'].sum().reindex([
                    'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'
                ])
                fig5 = px.pie(values=weekday_sum.values, names=weekday_sum.index, title="Sales by Weekday",
                              color_discrete_sequence=px.colors.sequential.Teal)
                st.plotly_chart(fig5, use_container_width=True)

                # Forecast Table
                st.subheader("Forecast Table")
                st.dataframe(fcst[["ds","yhat","yhat_lower","yhat_upper"]].tail(horizon))

            st.success("Forecast completed!")
        except Exception:
            st.error("Forecast failed. See traceback:")
            st.text(traceback.format_exc())

st.markdown("---")
st.caption("Professional BI-style dashboard with KPI tracking, multiple charts, and theme toggle.")
