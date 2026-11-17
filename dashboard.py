
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CarePulse | Nurse Demand Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

FORECAST_FILE = Path("outputs/demand_forecast.csv")
METRICS_FILE = Path("outputs/model_metrics.csv")

# --------------------------------------------------
# PROFESSIONAL UI STYLING
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #142B4A;
    --teal: #0D9488;
    --muted: #64748B;
    --border: #E2E8F0;
    --surface: #FFFFFF;
}

.stApp {
    background: #F5F8FC;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(245,248,252,0.95);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--navy) !important;
}

.hero {
    background: linear-gradient(120deg, #142B4A 0%, #174766 65%, #0D9488 100%);
    padding: 30px 32px;
    border-radius: 20px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 10px 28px rgba(20,43,74,0.13);
}

.hero h1 {
    color: white !important;
    font-size: 30px;
    font-weight: 800;
    margin: 0 0 8px 0;
}

.hero p {
    color: #DCEAF4;
    margin: 0;
    font-size: 14px;
}

.hero-tag {
    display: inline-block;
    padding: 6px 11px;
    margin-top: 17px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    color: #FFFFFF;
    font-size: 12px;
}

.section-heading {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 21px;
    color: #142B4A;
    margin: 25px 0 14px 0;
}

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 19px 19px 17px 19px;
    min-height: 128px;
    box-shadow: 0 3px 12px rgba(15,23,42,0.035);
}

.kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748B;
    margin-bottom: 12px;
}

.kpi-value {
    font-family: 'Manrope', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #142B4A;
    line-height: 1.2;
}

.kpi-note {
    font-size: 11px;
    color: #0D9488;
    margin-top: 9px;
}

.panel {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
}

.status-pill {
    display: inline-block;
    background: #CCFBF1;
    color: #115E59;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
}

div[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

div[data-testid="stSidebar"] h2 {
    font-size: 18px;
}

.stButton button, .stDownloadButton button {
    border-radius: 9px;
    font-weight: 600;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    padding: 25px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>CarePulse | Nurse Demand Intelligence</h1>
    <p>AI-powered demand forecasting and workforce planning</p>
    <span class="hero-tag">AI/ML Internship Project · Task AI-08</span>
    <span class="hero-tag">Synthetic demonstration data</span>
</div>
""", unsafe_allow_html=True)

if not FORECAST_FILE.exists() or not METRICS_FILE.exists():
    st.error(
        "Forecast files are missing. Run "
        "'python forecast.py' in the project terminal first."
    )
    st.stop()

forecast = pd.read_csv(FORECAST_FILE, parse_dates=["date"])
metrics = pd.read_csv(METRICS_FILE)

required_columns = {
    "date", "city", "service_type", "predicted_bookings"
}

if not required_columns.issubset(forecast.columns):
    st.error("The forecast CSV is missing required columns.")
    st.stop()

if forecast.empty:
    st.error("The forecast file contains no records.")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.markdown("## ⚙️ Forecast Controls")
st.sidebar.caption("Customize your staffing scenario.")

cities = sorted(forecast["city"].dropna().unique())

selected_city = st.sidebar.selectbox(
    "📍 Select city",
    cities,
)

available_services = sorted(
    forecast.loc[
        forecast["city"] == selected_city,
        "service_type",
    ].dropna().unique()
)

selected_services = st.sidebar.multiselect(
    "🩺 Care services",
    available_services,
    default=available_services,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### What-if analysis")

demand_change = st.sidebar.slider(
    "Demand adjustment",
    min_value=-50,
    max_value=100,
    value=0,
    step=5,
    format="%d%%",
)

st.sidebar.caption(
    "Increase or decrease the selected city's forecast "
    "to explore alternative staffing scenarios."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Staffing assumption")

bookings_per_nurse = st.sidebar.number_input(
    "Bookings per nurse per day",
    min_value=1,
    max_value=20,
    value=4,
    step=1,
)

st.sidebar.caption(
    "Illustrative planning assumption. "
    "Actual staffing depends on care duration, acuity, "
    "travel and shift arrangements."
)

# --------------------------------------------------
# FILTER AND CALCULATE
# --------------------------------------------------
city_data = forecast[
    (forecast["city"] == selected_city)
    & forecast["service_type"].isin(selected_services)
].copy()

if city_data.empty:
    st.warning("Select at least one service to view the forecast.")
    st.stop()

city_data["adjusted_demand"] = (
    city_data["predicted_bookings"].clip(lower=0)
    * (1 + demand_change / 100)
)

# Whole nurses required for each city/service/day.
city_data["nurses_needed"] = city_data[
    "adjusted_demand"
].apply(
    lambda value: math.ceil(
        max(0, value) / bookings_per_nurse
    )
)


daily = (
    city_data.groupby("date", as_index=False)
    .agg(
        Original_Demand=("predicted_bookings", "sum"),
        Adjusted_Demand=("adjusted_demand", "sum"),
    )
)

# Calculate nurses from total daily demand
daily["Nurses_Needed"] = daily["Adjusted_Demand"].apply(
    lambda value: math.ceil(
        max(0, value) / bookings_per_nurse
    )
)


total_bookings = city_data["adjusted_demand"].sum()
average_bookings = daily["Adjusted_Demand"].mean()
peak_bookings = daily["Adjusted_Demand"].max()
peak_nurses = daily["Nurses_Needed"].max()

original_total = city_data["predicted_bookings"].sum()
additional_bookings = total_bookings - original_total

start_date = city_data["date"].min().strftime("%d %b %Y")
end_date = city_data["date"].max().strftime("%d %b %Y")

# --------------------------------------------------
# TOP SUMMARY
# --------------------------------------------------
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;
    align-items:center;flex-wrap:wrap;gap:10px;margin:8px 0 20px 0;">
        <div>
            <div style="font-size:22px;font-weight:800;color:#142B4A;">
                Operations Overview
            </div>
            <div style="font-size:13px;color:#64748B;margin-top:4px;">
                {selected_city} · {start_date} — {end_date}
            </div>
        </div>
        <span class="status-pill">FORECAST READY</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Four custom KPI cards
kpis = [
    (
        "TOTAL FORECAST BOOKINGS",
        f"{total_bookings:,.0f}",
        f"{demand_change:+d}% demand scenario",
    ),
    (
        "DAILY AVERAGE",
        f"{average_bookings:,.1f}",
        "Bookings across selected services",
    ),
    (
        "PEAK DAILY DEMAND",
        f"{peak_bookings:,.1f}",
        "Highest forecast bookings in a day",
    ),
    (
        "PEAK NURSE REQUIREMENT",
        f"{peak_nurses:,.0f}",
        f"{bookings_per_nurse} bookings per nurse/day",
    ),
]

cols = st.columns(4)

for col, (label, value, note) in zip(cols, kpis):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Scenario summary
st.markdown("")
if demand_change != 0:
    st.info(
        f"Scenario for {selected_city}: "
        f"{demand_change:+d}% demand adjustment changes the "
        f"forecast by approximately {additional_bookings:+,.0f} "
        "bookings over the displayed period."
    )

# --------------------------------------------------
# TABBED DASHBOARD
# --------------------------------------------------
tab_overview, tab_staffing, tab_data, tab_model = st.tabs([
    "📊 Demand Overview",
    "👩‍⚕️ Staffing Plan",
    "📋 Forecast Data",
    "🧪 Model Evaluation",
])

# --------------------------------------------------
# TAB 1: DEMAND OVERVIEW
# --------------------------------------------------
with tab_overview:
    st.markdown(
        '<div class="section-heading">Demand Trend</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Compare the model's original forecast with your "
        "adjusted what-if scenario."
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        daily["date"],
        daily["Original_Demand"],
        color="#2563EB",
        linewidth=2.5,
        label="Original forecast",
    )
    ax.plot(
        daily["date"],
        daily["Adjusted_Demand"],
        color="#0D9488",
        linewidth=2.5,
        linestyle="--" if demand_change else "-",
        label="Adjusted scenario",
    )

    ax.fill_between(
        daily["date"],
        daily["Original_Demand"],
        alpha=0.07,
        color="#2563EB",
    )

    ax.set_ylabel("Bookings per day", color="#475569")
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#E2E8F0")
    ax.tick_params(colors="#64748B")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown(
            '<div class="section-heading">Bookings by Service</div>',
            unsafe_allow_html=True,
        )

        service_summary = (
            city_data.groupby("service_type", as_index=False)
            .agg(
                Forecast_Bookings=("predicted_bookings", "sum"),
                Adjusted_Bookings=("adjusted_demand", "sum"),
            )
            .sort_values("Adjusted_Bookings", ascending=True)
        )

        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.barh(
            service_summary["service_type"],
            service_summary["Adjusted_Bookings"],
            color="#0D9488",
            height=0.58,
        )
        ax2.set_xlabel("Bookings over forecast period")
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.spines["bottom"].set_color("#E2E8F0")
        ax2.tick_params(axis="y", length=0, colors="#334155")
        ax2.tick_params(axis="x", colors="#64748B")
        ax2.grid(axis="x", color="#E2E8F0", alpha=0.8)
        ax2.set_axisbelow(True)
        fig2.tight_layout()

        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with right:
        st.markdown(
            '<div class="section-heading">Service Snapshot</div>',
            unsafe_allow_html=True,
        )

        snapshot = (
            city_data.groupby("service_type", as_index=False)
            .agg(
                Bookings=("adjusted_demand", "sum"),
                Peak_Nurses=("nurses_needed", "max"),
            )
        )
        snapshot["Bookings"] = snapshot["Bookings"].round(0).astype(int)

        st.dataframe(
            snapshot.rename(columns={
                "service_type": "Care service",
                "Bookings": "Bookings",
                "Peak_Nurses": "Peak nurses",
            }),
            use_container_width=True,
            hide_index=True,
        )

# --------------------------------------------------
# TAB 2: STAFFING PLAN
# --------------------------------------------------
with tab_staffing:
    st.markdown(
        '<div class="section-heading">Daily Workforce Plan</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Nurse estimates use the selected bookings-per-nurse assumption. "
        "They are not an official staffing recommendation."
    )

    staffing = daily[
        ["date", "Adjusted_Demand", "Nurses_Needed"]
    ].copy()

    staffing = staffing.rename(columns={
        "date": "Date",
        "Adjusted_Demand": "Forecast bookings",
        "Nurses_Needed": "Estimated nurses",
    })

    staffing["Date"] = staffing["Date"].dt.strftime("%d %b %Y")
    staffing["Forecast bookings"] = staffing[
        "Forecast bookings"
    ].round(1)

    st.dataframe(
        staffing,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download staffing plan",
        data=staffing.to_csv(index=False).encode("utf-8"),
        file_name="nurse_staffing_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )

# --------------------------------------------------
# TAB 3: FORECAST DATA
# --------------------------------------------------
with tab_data:
    st.markdown(
        '<div class="section-heading">Forecast Records</div>',
        unsafe_allow_html=True,
    )

    details = city_data[[
        "date",
        "city",
        "service_type",
        "predicted_bookings",
        "adjusted_demand",
        "nurses_needed",
    ]].copy()

    details = details.rename(columns={
        "date": "Date",
        "city": "City",
        "service_type": "Care service",
        "predicted_bookings": "Original forecast",
        "adjusted_demand": "Adjusted demand",
        "nurses_needed": "Estimated nurses",
    })

    details["Date"] = details["Date"].dt.strftime("%Y-%m-%d")
    details["Original forecast"] = details["Original forecast"].round(2)
    details["Adjusted demand"] = details["Adjusted demand"].round(2)

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download forecast CSV",
        data=details.to_csv(index=False).encode("utf-8"),
        file_name="nurse_staffing_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

# --------------------------------------------------
# TAB 4: MODEL EVALUATION
# --------------------------------------------------
with tab_model:
    st.markdown(
        '<div class="section-heading">Forecast Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "The final 28 historical days are held out for testing. "
        "MAE and MAPE measure errors; lower values are generally better."
    )

    if {"city", "service_type"}.issubset(metrics.columns):
        city_metrics = metrics[
            (metrics["city"] == selected_city)
            & (metrics["service_type"].isin(selected_services))
        ].copy()
    else:
        city_metrics = pd.DataFrame()

    if not city_metrics.empty:
        st.dataframe(
            city_metrics,
            use_container_width=True,
            hide_index=True,
        )

        if {"model_MAE", "baseline_MAE"}.issubset(city_metrics.columns):
            model_mae = city_metrics["model_MAE"].mean()
            baseline_mae = city_metrics["baseline_MAE"].mean()

            if baseline_mae > 0:
                improvement = (
                    (baseline_mae - model_mae) / baseline_mae
                ) * 100

                st.metric(
                    "Mean MAE vs baseline",
                    f"{model_mae:.2f}",
                    delta=f"{improvement:.1f}% vs baseline",
                    delta_color="normal",
                )
            else:
                st.metric("Mean model MAE", f"{model_mae:.2f}")
    else:
        st.info("Select one or more services to view evaluation results.")

    st.warning(
        "Evaluation is performed on synthetic data. "
        "Results demonstrate the workflow and do not establish "
        "real-world forecasting accuracy."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    CAREPULSE · DEMAND FORECASTING FOR NURSE STAFFING<br>
    AI/ML Internship Assignment AI-08 · Synthetic data only
</div>
""", unsafe_allow_html=True)
