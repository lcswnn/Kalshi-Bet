import pandas as pd

df = pd.read_csv("weather_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")  # Sort chronologically

print(f"Data starts: {df['date'].min()}")
print(f"Data ends: {df['date'].max()}")
print(f"\nMost recent 5 days in data:")
print(df.tail(5)[["date", "value"]])