"""
Week 1 - Logistics Data Analyst Intern
Strategic Planning and Data Exploration

This prototype uses a small synthetic dataset only to demonstrate the
proposed analysis workflow. Replace it with the Week 2 cleaned dataset
before using the results for a real business decision.
"""

import pandas as pd

df = pd.read_csv("sample_logistics_data.csv")

# Basic exploration
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:\n", df.isna().sum())
print("\nSummary statistics:\n", df.describe(numeric_only=True))

# KPI 1: On-time delivery rate
on_time_rate = (df["On_Time_Rate"] >= 95).mean() * 100

# KPI 2: Average delivery delay
avg_delay_days = df["Delay_Days"].mean()

# KPI 3: Average transportation distance
avg_distance = df["Distance_km"].mean()

print(f"\nKPI - On-time shipment percentage: {on_time_rate:.2f}%")
print(f"KPI - Average delay: {avg_delay_days:.2f} days")
print(f"KPI - Average distance: {avg_distance:.2f} km")

# Regional comparison
regional = df.groupby("Region").agg(
    Shipments=("Shipment_ID", "count"),
    Avg_Delay_Days=("Delay_Days", "mean"),
    Avg_On_Time_Rate=("On_Time_Rate", "mean")
).sort_values("Avg_Delay_Days", ascending=False)

print("\nRegional performance:\n", regional)

# Simple predictive-model roadmap (not executed in Week 1):
# Target: Actual_Hours
# Features: Distance_km, Planned_Days, Mode, Priority, Region
# Candidate model: Linear Regression / Random Forest Regressor

# Clustering roadmap:
# Standardize numeric features and group shipments into operational
# segments such as low-risk, medium-risk, and high-risk shipments.

# Optimization roadmap:
# Formulate a route/carrier allocation problem that minimizes total
# transport cost and delay subject to vehicle capacity and delivery
# time-window constraints.
