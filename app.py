import traceback
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import calendar

# Prophet import (handle absence gracefully)
try:
    from prophet import Prophet
except Exception:
    Prophet = None

# ===== Page config (set early) =====
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ===== Custom Header UI/UX =====
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg, #1a2980, #26d0ce);
            padding: 1.2rem 2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
            font-family: 'Segoe UI', sans-serif;
        }
        .main-header h1 {
            font-size: 1.8rem;
            margin-bottom: 0.2rem;
            letter-spacing: 0.3px;
        }
        .main-header p {
            font-size: 0.95rem;
            color: #f0f0f0;
            margin: 0;
        }
        @media (max-width: 600px) {
            .main-header h1 { font-size: 1.2rem; }
            .main-header p { font-size: 0.85rem; }
        }
    </style>
    <div class="main-header">
        <h1>AI-Powered Sales Forecasting Dashboard</h1>
        <p>Track KPIs, visualize trends, forecast sales, and get insights for better decisions</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

# ===== Sidebar: Theme Toggle & Comparison Options =====
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"]) 
show_comparisons = st.sidebar.checkbox("Show Monthly & Yearly Comparisons", True)
monthly_agg_mode = st.sidebar.selectbox("Monthly aggregation", ["Sum", "Average"]) 
show_filters = st.sidebar.checkbox("Show Filters (Category/Store/Region)", True)
# New: Top items & low seasons controls
show_top_items = st.sidebar.checkbox("Show Top-selling Items", True)
top_k = st.sidebar.slider("Top K items", 3, 20, 5)
show_low_seasons = st.sidebar.checkbox("Show Low Seasons", True)
low_season_count = st.sidebar.slider("Low season months to highlight", 1, 6, 3)
st.sidebar.markdown("---")

# ===== Colors Based on Theme =====
if theme == "Dark":
    bg_color = "#0c1c3b"
    sidebar_color = "#111b2c"
    text_color = "#ffffff"
    table_bg1 = "#1a2a4c"
    table_bg2 = "#16203f"
    line_colors = ["#00ffcc", "#ff9900", "#00ccff", "#ff6600"]
    plotly_template = "plotly_dark"
else:
    bg_color = "#f5f5f5"
    sidebar_color = "#e0e0e0"
    text_color = "#0c1c3b"
    table_bg1 = "#ffffff"
    table_bg2 = "#e8e8e8"
    line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    plotly_template = "plotly_white"

# ===== Apply CSS =====
st.markdown(
    f"""
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
""",
    unsafe_allow_html=True,
)

# ===== Helpers =====
def make_sample_data(days=540):
    # create a longer sample dataset with category/store/region and item columns
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    base = 100 + pd.Series(range(days)) * 0.2
    seasonal = 10 * np.sin(np.arange(days) * 2 * np.pi / 30)
    trend = pd.Series(range(days)).rolling(7).mean().fillna(0)
    vals = (base + seasonal + trend).round(2)

    # categories, stores, regions, and items per category
    categories = ["Electronics", "Furniture", "Clothing"]
    stores = ["Store A", "Store B", "Store C"]
    regions = ["North", "South", "East", "West"]
    items = {
        "Electronics": ["Phone","Laptop","Headphones","Camera"],
        "Furniture": ["Chair","Table","Sofa","Shelf"],
        "Clothing": ["T-Shirt","Jeans","Jacket","Dress"],
    }

    rows = []
    # simulate multiple rows per date across categories/stores/items
    for i, d in enumerate(dates):
        for c in categories:
            for s in stores:
                region = np.random.choice(regions)
                for it in items[c]:
                    # variation by category/store/item
                    multiplier = 1.0
                    if c == "Electronics":
                        multiplier = 1.2
                    elif c == "Furniture":
                        multiplier = 0.9
                    item_mult = 1.0 + (hash(it) % 10 - 5) / 100.0
                    val = float(max(0.0, vals.iloc[i] * multiplier * item_mult * (1 + (np.random.rand()-0.5) * 0.15)))
                    rows.append({"ds": d, "y": round(val, 2), "category": c, "store": s, "region": region, "item": it})
    return pd.DataFrame(rows)


@st.cache_data
def get_data(path="data/sales.csv"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(p, parse_dates=[0])
    # normalize expected column names
    if "date" in df.columns and "ds" not in df.columns:
        df = df.rename(columns={"date": "ds"})
    if "sales" in df.columns and "y" not in df.columns:
        df = df.rename(columns={"sales": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    # ensure optional columns exist (category/store/region/item)
    if "category" not in df.columns:
        df["category"] = "All"
    if "store" not in df.columns:
        df["store"] = "All"
    if "region" not in df.columns:
        df["region"] = "All"
    if "item" not in df.columns:
        # try 'product' fallback
        if "product" in df.columns:
            df = df.rename(columns={"product": "item"})
        else:
            df["item"] = "Unknown Item"

    return df[["ds", "y", "category", "store", "region", "item"]].sort_values("ds").reset_index(drop=True)

# ===== Sidebar: Inputs =====
uploaded = st.sidebar.file_uploader("Upload CSV (must include ds/date, sales/y; optional: category, store, region, item)", type=["csv"])
model_choice = st.sidebar.selectbox("Model", ["Prophet"]) 
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, 28)
st.sidebar.markdown("---")

# ===== Load Data =====
df = None
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        # flexible parsing
        if "date" in df.columns and "ds" not in df.columns:
            df = df.rename(columns={"date": "ds"})
        if "sales" in df.columns and "y" not in df.columns:
            df = df.rename(columns={"sales": "y"})
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
        else:
            st.sidebar.error("Uploaded CSV missing 'date'/'ds' column. Using sample data.")
            df = make_sample_data()

        # optional columns
        if "category" not in df.columns:
            df["category"] = "All"
        if "store" not in df.columns:
            df["store"] = "All"
        if "region" not in df.columns:
            df["region"] = "All"
        if "item" not in df.columns and "product" in df.columns:
            df = df.rename(columns={"product": "item"})
        if "item" not in df.columns:
            df["item"] = "Unknown Item"

        df = df[["ds", "y", "category", "store", "region", "item"]].sort_values("ds").reset_index(drop=True)
    except Exception:
        st.sidebar.error("CSV parse error. Using sample data.")
        df = make_sample_data()
else:
    try:
        df = get_data("data/sales.csv")
    except Exception:
        df = make_sample_data()

# ===== Filters: Category / Store / Region =====
if show_filters:
    st.sidebar.markdown("### Filters")
    unique_categories = list(pd.Index(df["category"]).unique())
    unique_stores = list(pd.Index(df["store"]).unique())
    unique_regions = list(pd.Index(df["region"]).unique())

    sel_categories = st.sidebar.multiselect("Category", options=unique_categories, default=unique_categories)
    sel_stores = st.sidebar.multiselect("Store", options=unique_stores, default=unique_stores)
    sel_regions = st.sidebar.multiselect("Region", options=unique_regions, default=unique_regions)
else:
    # defaults: include all
    sel_categories = list(pd.Index(df["category"]).unique())
    sel_stores = list(pd.Index(df["store"]).unique())
    sel_regions = list(pd.Index(df["region"]).unique())

# Apply filters
filtered_df = df[
    (df["category"].isin(sel_categories)) &
    (df["store"].isin(sel_stores)) &
    (df["region"].isin(sel_regions))
].reset_index(drop=True)

# If filters remove all data, show warning and revert to original
if filtered_df.empty:
    st.warning("No data matches selected filters — showing full dataset instead.")
    filtered_df = df.copy()

# Use filtered_df for all downstream calculations
working_df = filtered_df.copy()

# ===== KPI Tracking =====
working_df["daily_change"] = working_df["y"].diff()
working_df["weekly_sum"] = working_df["y"].rolling(7).sum()
working_df["cum_sales"] = working_df["y"].cumsum()
working_df["trend_7d"] = working_df["y"].rolling(window=7, center=True, min_periods=1).mean()

# Add year/month columns for comparisons
working_df["year"] = working_df["ds"].dt.year
working_df["month_num"] = working_df["ds"].dt.month
working_df["month_name"] = working_df["ds"].dt.month_name()

# Basic single-value KPIs
total_sales = working_df["y"].sum()
avg_sales = working_df["y"].mean()
last_sale = working_df["y"].iloc[-1]
max_sale = working_df["y"].max()
min_sale = working_df["y"].min()
daily_change = working_df["daily_change"].iloc[-1] if len(working_df) > 1 else 0
weekly_growth = (
    working_df["weekly_sum"].iloc[-1] - working_df["weekly_sum"].iloc[-8]
    if len(working_df) >= 8
    else working_df["weekly_sum"].iloc[-1] if len(working_df) >= 1 else 0
)

# ===== Current period KPIs (monthly & yearly comparisons) =====
last_date = working_df["ds"].max()
current_year = int(last_date.year)
current_month = int(last_date.month)

# Monthly totals
monthly_agg = working_df.groupby(["year", "month_num"]) ["y"].agg("sum" if monthly_agg_mode == "Sum" else "mean").reset_index()

def get_month_value(year, month):
    r = monthly_agg[(monthly_agg["year"] == year) & (monthly_agg["month_num"] == month)]
    if r.empty:
        return 0.0
    return float(r["y"].values[0])

curr_month_val = get_month_value(current_year, current_month)
prev_month = current_month - 1
prev_month_year = current_year
if prev_month == 0:
    prev_month = 12
    prev_month_year -= 1
prev_month_val = get_month_value(prev_month_year, prev_month)
month_change_pct = ((curr_month_val - prev_month_val) / prev_month_val * 100) if prev_month_val != 0 else None

# Yearly totals and YoY
yearly_agg = working_df.groupby("year")["y"].sum().reset_index()
last_year_val = float(yearly_agg[yearly_agg["year"] == current_year]["y"].values[0]) if current_year in yearly_agg["year"].values else 0.0
prev_year_val = float(yearly_agg[yearly_agg["year"] == (current_year - 1)]["y"].values[0]) if (current_year - 1) in yearly_agg["year"].values else 0.0
yoy_pct = ((last_year_val - prev_year_val) / prev_year_val * 100) if prev_year_val != 0 else None

# ===== Top-selling items =====
# compute top items if item column exists
if "item" in working_df.columns and show_top_items:
    item_agg = working_df.groupby("item")["y"].sum().reset_index().sort_values("y", ascending=False)
    item_agg["rank"] = range(1, len(item_agg) + 1)
    top_items = item_agg.head(top_k)
else:
    item_agg = pd.DataFrame(columns=["item","y","rank"]) 
    top_items = pd.DataFrame(columns=["item","y","rank"]) 

# ===== Low seasons detection (months with lowest avg sales) =====
# compute average sales per calendar month across all years
low_seasons = []
monthly_avg = working_df.groupby("month_num")["y"].mean().reset_index()
monthly_avg["month_name"] = monthly_avg["month_num"].apply(lambda x: calendar.month_name[int(x)])
monthly_avg_sorted = monthly_avg.sort_values("y")
low_seasons = monthly_avg_sorted.head(low_season_count)[["month_num","month_name","y"]]

# Also prepare heatmap data (years x months) for seasonality
season_pivot = working_df.groupby(["year","month_num"])["y"].sum().reset_index()
season_pivot = season_pivot.pivot(index="year", columns="month_num", values="y").fillna(0)
season_pivot.columns = [calendar.month_name[int(c)] for c in season_pivot.columns]

# ===== Render UI =====
st.markdown("<h1>Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)

# show active filters summary
with st.expander("Active filters summary", expanded=False):
    st.write(f"Categories: {', '.join(sel_categories)}")
    st.write(f"Stores: {', '.join(sel_stores)}")
    st.write(f"Regions: {', '.join(sel_regions)}")

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)
kpi1.metric("Total Sales", f"{total_sales:,.0f}")
kpi2.metric("Avg Daily Sales", f"{avg_sales:.2f}")
kpi3.metric("Latest Sale", f"{last_sale:.0f}")
kpi4.metric("Max Sale", f"{max_sale:.0f}")
kpi5.metric("Min Sale", f"{min_sale:.0f}")
kpi6.metric("Daily Change", f"{daily_change:+.0f}")
kpi7.metric("Weekly Growth", f"{weekly_growth:+.0f}")

st.markdown("---")

# ===== Insight cards for decision-making =====
# compute actionable insights (safe guards for small datasets)
try:
    insights_df = working_df.copy()
    insight_notes = []

    # Top store
    if "store" in insights_df.columns and len(insights_df) > 0:
        top_store_row = insights_df.groupby("store")["y"].sum().reset_index().sort_values("y", ascending=False).head(1)
        top_store = top_store_row.iloc[0]["store"]
        top_store_val = float(top_store_row.iloc[0]["y"])
    else:
        top_store = "N/A"
        top_store_val = 0.0

    # Top category
    if "category" in insights_df.columns and len(insights_df) > 0:
        top_cat_row = insights_df.groupby("category")["y"].sum().reset_index().sort_values("y", ascending=False).head(1)
        top_category = top_cat_row.iloc[0]["category"]
        top_category_val = float(top_cat_row.iloc[0]["y"])
    else:
        top_category = "N/A"
        top_category_val = 0.0

    # Low season month (from monthly_avg computed earlier)
    if "monthly_avg" in locals() and not monthly_avg.empty:
        worst_month_row = monthly_avg.sort_values("y").head(1).iloc[0]
        low_season_month = f"{int(worst_month_row['month_num'])} - {worst_month_row['month_name']}"
    else:
        low_season_month = "N/A"

    # 30-day growth vs previous 30-day period
    insights_df_sorted = insights_df.sort_values("ds").reset_index(drop=True)
    if len(insights_df_sorted) >= 60:
        last_30 = insights_df_sorted.tail(30)["y"].sum()
        prev_30 = insights_df_sorted.tail(60).head(30)["y"].sum()
        growth_30_pct = ((last_30 - prev_30) / prev_30 * 100) if prev_30 != 0 else None
    else:
        growth_30_pct = None

    # Stores with declining trend (last 90 vs previous 90 days)
    declining_stores = []
    try:
        if "store" in insights_df.columns and len(insights_df_sorted) >= 180:
            stores = insights_df_sorted["store"].unique()
            for s in stores:
                s_df = insights_df_sorted[insights_df_sorted["store"] == s]
                if len(s_df) >= 180:
                    last_90 = s_df.tail(90)["y"].sum()
                    prev_90 = s_df.tail(180).head(90)["y"].sum()
                    pct = ((last_90 - prev_90) / prev_90 * 100) if prev_90 != 0 else 0
                    if pct < -5:  # arbitrary threshold
                        declining_stores.append((s, pct))
    except Exception:
        declining_stores = []

except Exception:
    top_store = "N/A"
    top_store_val = 0.0
    top_category = "N/A"
    top_category_val = 0.0
    low_season_month = "N/A"
    growth_30_pct = None
    declining_stores = []

st.subheader("Actionable Insights")
card1, card2, card3, card4 = st.columns(4)
card1.metric("Top Store", f"{top_store}", f"{top_store_val:,.0f}")
card2.metric("Top Category", f"{top_category}", f"{top_category_val:,.0f}")
if growth_30_pct is None:
    card3.metric("30-day Growth", "Insufficient data", delta=None)
else:
    card3.metric("30-day Growth", f"{growth_30_pct:.2f}%", delta=f"{growth_30_pct:.2f}%")
card4.metric("Low Season Month", low_season_month)

# Quick recommendation box
with st.expander("Recommendations & Next Actions", expanded=False):
    if top_store != "N/A":
        st.write(f"**Focus inventory & promotions on {top_store}** — it has the highest sales ({top_store_val:,.0f}).")
    if top_category != "N/A":
        st.write(f"**Promote top category — {top_category}** across channels and highlight best items in that category.")
    if growth_30_pct is not None and growth_30_pct < 0:
        st.warning(f"Sales declined by {growth_30_pct:.2f}% over the last 30 days compared to the previous 30 days. Consider targeted promotions or inventory checks.")
    if declining_stores:
        st.error("Stores with meaningful decline (last 90d vs prev 90d):")
        for s, pct in declining_stores:
            st.write(f"- {s}: {pct:.1f}%")
    st.info("Other tactics: run A/B promo tests on top categories, bundle low-season products with high-demand items, re-balance inventory to fastest-moving SKUs.")

st.markdown("---")
st.subheader("Recent Sales Data")
st.dataframe(working_df.tail(10))

# ===== Top-selling items section =====
if show_top_items:
    st.markdown("---")
    st.subheader(f"Top-selling Items (Top {top_k})")
    if not top_items.empty:
        # bar chart
        fig_top = px.bar(
            top_items,
            x="item",
            y="y",
            title=f"Top {top_k} Items by Sales",
            labels={"y": "Total Sales", "item": "Item"},
            template=plotly_template,
        )
        st.plotly_chart(fig_top, use_container_width=True)

        # show table
        st.dataframe(top_items.reset_index(drop=True))

        # highlight best-selling item as KPI
        best_item = top_items.iloc[0]
        st.info(f"Top item: {best_item['item']} — {best_item['y']:,.0f} total sales")
    else:
        st.write("No item-level data available in the dataset.")

# ===== Low seasons section =====
if show_low_seasons:
    st.markdown("---")
    st.subheader("Low Seasons (Months with lowest average sales)")
    if not low_seasons.empty:
        # show low seasons list
        for i, r in low_seasons.iterrows():
            st.write(f"{int(r['month_num'])} — {r['month_name']}: avg sales {r['y']:.2f}")

        # seasonality heatmap (years x months)
        fig_heat = px.imshow(
            season_pivot,
            labels=dict(x="Month", y="Year", color="Sales"),
            x=season_pivot.columns,
            y=season_pivot.index,
            title="Seasonality Heatmap (Sales by Year and Month)",
            aspect="auto",
            template=plotly_template,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.write("Not enough data to compute low seasons.")

# ===== Monthly & Yearly Comparisons Section =====
if show_comparisons:
    st.markdown("---")
    st.subheader("Monthly & Yearly Comparisons")

    # Monthly comparison: build pivot table months x years
    month_order = list(range(1, 13))
    month_name_order = [calendar.month_name[m] for m in month_order]

    if monthly_agg_mode == "Sum":
        agg_mode_name = "Monthly Sum"
        m_agg = working_df.groupby(["year", "month_num"]) ["y"].sum().reset_index()
    else:
        agg_mode_name = "Monthly Average"
        m_agg = working_df.groupby(["year", "month_num"]) ["y"].mean().reset_index()

    m_agg["month_name"] = m_agg["month_num"].apply(lambda x: calendar.month_name[int(x)])
    pivot = m_agg.pivot(index="month_num", columns="year", values="y").reindex(index=month_order).fillna(0)
    pivot = pivot.reset_index()
    pivot["month_name"] = pivot["month_num"].apply(lambda x: calendar.month_name[int(x)])

    melted = pivot.melt(id_vars=["month_num", "month_name"], var_name="year", value_name="value")

    fig_monthly = px.bar(
        melted,
        x="month_name",
        y="value",
        color="year",
        barmode="group",
        title=f"{agg_mode_name} by Month — Comparison Across Years",
        category_orders={"month_name": month_name_order},
        labels={"value": "Sales", "month_name": "Month", "year": "Year"},
        template=plotly_template,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    fig_yearly = px.bar(
        yearly_agg.sort_values("year"),
        x="year",
        y="y",
        title="Total Sales by Year",
        labels={"y": "Total Sales", "year": "Year"},
        template=plotly_template,
    )
    yearly_agg = yearly_agg.sort_values("year").reset_index(drop=True)
    yearly_agg["yoy_pct"] = yearly_agg["y"].pct_change() * 100
    fig_yearly.update_traces(text=yearly_agg["y"].map(lambda v: f"{v:,.0f}"), textposition="outside")
    st.plotly_chart(fig_yearly, use_container_width=True)

    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
    comp_col1.metric(f"{calendar.month_name[current_month]} {current_year}", f"{curr_month_val:,.0f}")
    if prev_month_val is not None:
        prev_label = f"{calendar.month_name[prev_month]} {prev_month_year}"
        if month_change_pct is None:
            comp_col2.metric(prev_label, f"{prev_month_val:,.0f}", delta=None)
        else:
            comp_col2.metric(prev_label, f"{prev_month_val:,.0f}", delta=f"{month_change_pct:.2f}%")
    else:
        comp_col2.metric("Previous Month", f"{prev_month_val:,.0f}", delta=None)

    comp_col3.metric(f"Year {current_year}", f"{last_year_val:,.0f}")
    if yoy_pct is None:
        comp_col4.metric(f"Year {current_year-1}", f"{prev_year_val:,.0f}", delta=None)
    else:
        comp_col4.metric(f"YoY Change", f"{prev_year_val:,.0f}", delta=f"{yoy_pct:.2f}%")

# ===== Forecast & Plots =====
if st.button("Run Forecast"):
    with st.spinner("Forecasting..."):
        try:
            if Prophet is None:
                st.error("Prophet not installed. Install 'prophet' package to run forecasting.")
            else:
                train_df = working_df[["ds", "y"]].dropna().reset_index(drop=True)

                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                m.fit(train_df)

                future = m.make_future_dataframe(periods=horizon)
                fcst = m.predict(future)

                fcst_small = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]

                merged = pd.merge(fcst_small, working_df[["ds", "y", "trend_7d"]], on="ds", how="left")

                fig_trend = go.Figure()
                actuals = merged.dropna(subset=["y"]) 
                fig_trend.add_trace(
                    go.Scatter(
                        x=actuals["ds"],
                        y=actuals["y"],
                        mode="markers",
                        name="Actual",
                        marker=dict(size=6),
                    )
                )
                trend_vals = merged.dropna(subset=["trend_7d"])
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_vals["ds"],
                        y=trend_vals["trend_7d"],
                        mode="lines",
                        name="7-day Trend (Actual)",
                        line=dict(width=3, dash="dot"),
                    )
                )
                fig_trend.add_trace(
                    go.Scatter(
                        x=merged["ds"],
                        y=merged["yhat"],
                        mode="lines",
                        name="Forecast (yhat)",
                        line=dict(width=2),
                    )
                )
                fig_trend.add_trace(
                    go.Scatter(
                        x=pd.concat([merged["ds"], merged["ds"][::-1]]),
                        y=pd.concat([merged["yhat_upper"], merged["yhat_lower"][::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,0,0,0.05)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=True,
                        name="Forecast Uncertainty",
                    )
                )
                fig_trend.update_layout(
                    title=f"Sales Trend — Actual vs Forecast (Next {horizon} days)",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    template=plotly_template,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=60, b=20, l=40, r=20),
                )

                st.plotly_chart(fig_trend, use_container_width=True)

                # Additional charts (kept from original) - use working_df
                fig1 = px.line(
                    merged,
                    x="ds",
                    y="yhat",
                    title=f"Forecast ({horizon} Days)",
                    labels={"yhat": "Forecast", "ds": "Date"},
                    template=plotly_template,
                )
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = px.line(
                    merged,
                    x="ds",
                    y=["y", "yhat"],
                    labels={"value": "Sales", "variable": "Series"},
                    template=plotly_template,
                )
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.histogram(
                    working_df,
                    x="y",
                    nbins=20,
                    title="Sales Distribution",
                    template=plotly_template,
                )
                st.plotly_chart(fig3, use_container_width=True)

                fig4 = px.area(
                    working_df,
                    x="ds",
                    y="cum_sales",
                    title="Cumulative Sales",
                    template=plotly_template,
                )
                st.plotly_chart(fig4, use_container_width=True)

                working_df["weekday"] = working_df["ds"].dt.day_name()
                weekday_sum = (
                    working_df.groupby("weekday")["y"].sum().reindex(
                        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    )
                )
                fig5 = px.pie(
                    values=weekday_sum.values,
                    names=weekday_sum.index,
                    title="Sales by Weekday",
                    template=plotly_template,
                )
                st.plotly_chart(fig5, use_container_width=True)

                st.subheader("Forecast Table")
                st.dataframe(merged[["ds", "y", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))

                st.success("Forecast completed!")
        except Exception:
            st.error("Forecast failed. See traceback:")
            st.text(traceback.format_exc())

st.markdown("---")
st.caption("Professional BI-style dashboard with KPI tracking, monthly & yearly comparisons, filters by category/store/region, highlights for top-selling items & low seasons, multiple charts, and theme toggle.")
