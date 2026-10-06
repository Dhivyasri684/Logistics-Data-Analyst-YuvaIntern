import pandas as pd
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "data/logistics_cleaned_data.csv"

df = pd.read_csv(INPUT_FILE)

# Example numerical normalization
numeric_cols = [
    "Quantity",
    "Shipping_Cost_INR",
    "Shipping_Time_Days",
    "Inventory_Level"
]

scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=[f"{c}_Normalized" for c in numeric_cols]
)

# Example categorical encoding
encoded = pd.get_dummies(
    df[["Product_Category", "Transport_Mode", "Delivery_Status"]],
    prefix=["Category", "Mode", "Status"],
    dtype=int
)

preprocessed = pd.concat([df, normalized, encoded], axis=1)
preprocessed.to_csv("data/logistics_preprocessed_data.csv", index=False)

print("Preprocessing completed.")
