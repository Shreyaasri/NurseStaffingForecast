
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# File paths
DATA_FILE = Path("data/daily_bookings.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values(["city", "service_type", "date"])

# Calendar features
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month
df["day_number"] = (df["date"] - df["date"].min()).dt.days

# Feature columns used by the model
FEATURES = [
    "day_of_week",
    "day_of_year",
    "month",
    "day_number",
    "trend_factor",
    "weekly_factor",
    "monsoon_factor",
    "festival_factor",
]

# Forecast settings
FORECAST_DAYS = 28
records = []
metrics = []

for (city, service), group in df.groupby(
    ["city", "service_type"]
):
    group = group.sort_values("date").copy()

    # Last 28 days are held out for testing
    train = group.iloc[:-FORECAST_DAYS]
    test = group.iloc[-FORECAST_DAYS:]

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(train[FEATURES], train["bookings"])

    # Evaluate model on the held-out period
    test_predictions = np.maximum(
        0, model.predict(test[FEATURES])
    )

    # Mean absolute percentage error, ignoring zero actuals
    actual = test["bookings"].to_numpy()
    nonzero = actual > 0

    if nonzero.any():
        mape = np.mean(
            np.abs(
                (actual[nonzero] - test_predictions[nonzero])
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    mae = mean_absolute_error(actual, test_predictions)

    # Simple baseline: repeat the previous week's demand
    baseline_predictions = (
        group["bookings"]
        .iloc[-FORECAST_DAYS * 2:-FORECAST_DAYS]
        .to_numpy()
    )

    baseline_mae = mean_absolute_error(
        actual, baseline_predictions
    )

    metrics.append({
        "city": city,
        "service_type": service,
        "model_MAE": round(mae, 2),
        "model_MAPE_percent": round(mape, 2),
        "baseline_MAE": round(baseline_mae, 2),
    })

    # Fit a fresh model on all available history
    model.fit(group[FEATURES], group["bookings"])

    # Prepare the next 28 days
    last_date = group["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D",
    )

    future = pd.DataFrame({"date": future_dates})
    future["day_of_week"] = future["date"].dt.dayofweek
    future["day_of_year"] = future["date"].dt.dayofyear
    future["month"] = future["date"].dt.month
    future["day_number"] = (
        future["date"] - df["date"].min()
    ).dt.days

    # Recreate assumed trend and seasonal factors
    future["trend_factor"] = 1 + (
        0.15 * future["day_number"] / 730
    )
    future["weekly_factor"] = np.where(
        future["day_of_week"].isin([5, 6]), 1.12, 1.0
    )
    future["monsoon_factor"] = np.where(
        future["month"].isin([6, 7, 8, 9]), 1.25, 1.0
    )

    # Recreate the synthetic festival assumptions
    festival_dates = pd.to_datetime([
        "2026-03-04", "2026-03-20",
        "2026-11-08", "2026-11-09",
    ])

    future["festival_factor"] = future["date"].apply(
        lambda d: 1.20
        if np.any(np.abs((festival_dates - d).days) <= 1)
        else 1.0
    )

    predictions = np.maximum(
        0, model.predict(future[FEATURES])
    )

    for date, prediction in zip(
        future["date"], predictions
    ):
        records.append({
            "date": date,
            "city": city,
            "service_type": service,
            "predicted_bookings": round(float(prediction), 2),
        })

# Save forecast and evaluation results
forecast_df = pd.DataFrame(records)
metrics_df = pd.DataFrame(metrics)

forecast_df.to_csv(
    OUTPUT_DIR / "demand_forecast.csv", index=False
)
metrics_df.to_csv(
    OUTPUT_DIR / "model_metrics.csv", index=False
)

print("Forecasting completed successfully!")
print("Forecast rows:", len(forecast_df))
print("Forecast saved to: outputs/demand_forecast.csv")
print("Metrics saved to: outputs/model_metrics.csv")
print("\nModel evaluation:")
print(metrics_df.to_string(index=False))
print("\nForecast preview:")
print(forecast_df.head(10).to_string(index=False))
