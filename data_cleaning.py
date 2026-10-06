import pandas as pd
import numpy as np

RAW_FILE = "data/logistics_raw_data.csv"
OUTPUT_FILE = "data/logistics_cleaned_data.csv"

df = pd.read_csv(RAW_FILE)

# Standardize text fields
for col in ["Product_Category", "Origin", "Destination", "Transport_Mode", "Delivery_Status"]:
    df[col] = df[col].astype("string").str.strip()

df["Product_Category"] = df["Product_Category"].str.title()
df["Transport_Mode"] = df["Transport_Mode"].str.title()

# Remove duplicate rows
df = df.drop_duplicates()

# Convert numeric columns and invalidate impossible values
numeric_cols = ["Quantity", "Shipping_Cost_INR", "Shipping_Time_Days", "Inventory_Level"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
df.loc[df["Shipping_Cost_INR"] <= 0, "Shipping_Cost_INR"] = np.nan
df.loc[df["Shipping_Time_Days"] <= 0, "Shipping_Time_Days"] = np.nan
df.loc[df["Inventory_Level"] < 0, "Inventory_Level"] = np.nan

# IQR capping for outliers
def iqr_cap(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)

for col in numeric_cols:
    df[col] = iqr_cap(df[col])

# Median imputation
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Date standardization
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Order_Date"] = df["Order_Date"].fillna(df["Order_Date"].mode()[0])

df.to_csv(OUTPUT_FILE, index=False)
print("Cleaning completed:", OUTPUT_FILE)
