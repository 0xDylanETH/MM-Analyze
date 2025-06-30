import pandas as pd

# Load merged data (using a relative path to the project root)
df = pd.read_pickle("../data/merged_df.pkl")

# Print the first few rows (or df.head())
print("First few rows of merged data:")
print(df.head())

# (Optional) Example: compute mean "Close" price per symbol
mean_close = df.groupby("symbol")["Close"].mean()
print("\nMean Close price (per symbol):")
print(mean_close)