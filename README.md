# NurseStaffingForecast 🏥

### AI/ML Internship Assignment — AI-08 | TheCarePal

An AI/ML-based nurse staffing demand forecasting project that predicts daily care-service bookings using synthetic data and presents demand forecasts, staffing estimates, and model evaluation through an interactive Streamlit dashboard.

## 📌 Project Overview

NurseStaffingForecast helps explore how historical booking patterns can be used to forecast future demand for care services and estimate nursing requirements.

The project uses synthetic booking data for three cities and four care services. A machine learning model forecasts demand for the next 28 days, while an interactive dashboard allows users to explore forecasts and staffing scenarios.

## 🎯 Objectives

- Generate synthetic historical booking data.
- Forecast demand for the next 28 days.
- Estimate daily nurse requirements from forecast bookings.
- Compare model performance against a baseline.
- Visualize forecasts and staffing plans in an interactive dashboard.
- Explore alternative demand scenarios using what-if analysis.

## ✨ Key Features

- **Demand Forecasting:** Predicts daily bookings for the selected city and care services.
- **Interactive Dashboard:** Displays KPIs, demand trends, and forecast records.
- **Staffing Plan:** Estimates nurse requirements using a configurable bookings-per-nurse assumption.
- **What-if Analysis:** Adjusts forecast demand to explore alternative scenarios.
- **Model Evaluation:** Reports MAE, MAPE, and baseline MAE.
- **CSV Downloads:** Allows users to download forecast and staffing-plan data.

## 🏙️ Cities Covered

- Indore
- Bhopal
- Jabalpur

## 🩺 Care Services

- Baby Care
- Dementia Care
- Elderly Care
- Post-Hospital Recovery

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## 📁 Project Structure

```text
NurseStaffingForecast/
├── data/
│   └── daily_bookings.csv
├── outputs/
│   ├── demand_forecast.csv
│   └── model_metrics.csv
├── generate_data.py
├── forecast.py
├── dashboard.py
├── README.md
└── requirements.txt
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shreyaasri/NurseStaffingForecast.git
cd NurseStaffingForecast
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

### 4. Generate the dataset

```bash
python generate_data.py
```

### 5. Generate forecasts and model evaluation results

```bash
python forecast.py
```

### 6. Launch the dashboard

```bash
python -m streamlit run dashboard.py
```

The dashboard will open in your browser.

## 📊 Model Evaluation

The project evaluates forecasting performance using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted bookings.
- **MAPE (Mean Absolute Percentage Error):** Measures percentage error for non-zero actual values.
- **Baseline MAE:** Provides a reference error for comparison with the forecasting model.

Lower error values generally indicate better predictions. Performance depends on the generated data and the evaluation period.

## 📈 Dashboard Sections

1. **Demand Overview:** Forecast trends and key demand indicators.
2. **Staffing Plan:** Daily forecast bookings and estimated nurse requirements.
3. **Forecast Data:** Detailed forecast records with CSV download.
4. **Model Evaluation:** Model error metrics and baseline comparison.

## ⚠️ Assumptions and Limitations

- The project uses synthetic demonstration data, not real patient or booking records.
- Forecasts demonstrate an ML workflow and do not establish real-world forecasting accuracy.
- Nurse estimates use an illustrative bookings-per-nurse assumption.
- Actual staffing decisions require consideration of care duration, patient needs, staff availability, and operational policies.

## 🚀 Future Improvements

- Integrate real historical booking data.
- Compare additional forecasting algorithms.
- Improve festival and seasonal demand features.
- Add automated model monitoring and retraining.
- Incorporate care duration and shift-level staffing requirements.
- Deploy the dashboard for online access.

## 👩‍💻 Author

**Shreya Kumari**

AI/ML Internship Assignment — AI-08, TheCarePal

---

*This project was developed for educational and internship demonstration purposes.*
