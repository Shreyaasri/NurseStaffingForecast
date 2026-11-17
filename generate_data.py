
import pandas as pd
import numpy as np
from pathlib import Path

# Make results reproducible
rng = np.random.default_rng(42)

# Project settings
CITIES = ["Indore", "Bhopal", "Jabalpur"]

SERVICES = [
    "Elderly Care",
    "Dementia Care",
    "Baby Care",
    "Post-Hospital Recovery",
]


# Two years of daily synthetic data ending today
end_date = pd.Timestamp.today().normalize()
start_date = end_date - pd.DateOffset(years=2) + pd.Timedelta(days=1)

dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D",
)


# Approximate festival dates for synthetic demand effects
festival_dates = pd.to_datetime([
    "2024-03-25",  # Holi
    "2024-04-11",  # Eid (approximate)
    "2024-11-01",  # Diwali
    "2025-03-14",  # Holi
    "2025-03-31",  # Eid (approximate)
    "2025-10-20",  # Diwali (approximate)
])

# Base daily demand for each service
base_demand = {
    "Elderly Care": 18,
    "Dementia Care": 8,
    "Baby Care": 14,
    "Post-Hospital Recovery": 12,
}

city_multiplier = {
    "Indore": 1.0,
    "Bhopal": 0.9,
    "Jabalpur": 0.7,
}

records = []
total_days = len(dates)

for city in CITIES:
    for service in SERVICES:
        for day_index, date in enumerate(dates):

            # Gradual demand growth over two years
            trend = 1 + 0.15 * day_index / (total_days - 1)

            # Weekly pattern: demand varies by day of week
            weekday = date.dayofweek
            weekly_factor = (
                1.12 if weekday in [5, 6] else 1.0
            )

            # Monsoon-related demand increase: June to September
            monsoon_factor = (
                1.25
                if date.month in [6, 7, 8, 9]
                else 1.0
            )

            # Festival effect: nearby dates receive a demand boost
            festival_factor = 1.0
            days_from_festival = abs(
                (festival_dates - date).days
            )

            if np.any(days_from_festival <= 1):
                festival_factor = 1.20

            # Combine the demand factors
            expected_demand = (
                base_demand[service]
                * city_multiplier[city]
                * trend
                * weekly_factor
                * monsoon_factor
                * festival_factor
            )

            # Add realistic random variation
            bookings = int(rng.poisson(expected_demand))

            records.append({
                "date": date,
                "city": city,
                "service_type": service,
                "bookings": bookings,
                "trend_factor": round(trend, 3),
                "weekly_factor": weekly_factor,
                "monsoon_factor": monsoon_factor,
                "festival_factor": festival_factor,
            })

# Create the dataset
df = pd.DataFrame(records)

# Save output in the project folder
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "daily_bookings.csv"
df.to_csv(output_file, index=False)

# Display a quick summary
print("Dataset created successfully!")
print("Saved to:", output_file)
print("Total rows:", len(df))
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Cities:", df["city"].nunique())
print("Service types:", df["service_type"].nunique())
print("\nFirst five rows:")
print(df.head())
