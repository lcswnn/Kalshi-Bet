import pandas as pd
import re

df = pd.read_csv("kalshi_chicago_weather.csv", names=[
    "ticker", "title", "subtitle", "event_ticker", "close_time", 
    "expiration_time", "result", "last_price", "volume", "yes_bid", "yes_ask"
])

def parse_event_date(event_ticker):
    parts = event_ticker.split("-")
    if len(parts) >= 2:
        date_part = parts[1]
        year = "20" + date_part[:2]
        month_str = date_part[2:5]
        day = date_part[5:]
        months = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
                  "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12"}
        month = months.get(month_str, "01")
        return f"{year}-{month}-{day}"
    return None

df["date"] = df["event_ticker"].apply(parse_event_date)

winners = df[df["result"] == "yes"].copy()

def extract_temp(subtitle):
    # Find all numbers in the string
    numbers = re.findall(r'\d+', str(subtitle))
    if len(numbers) >= 2:
        # Band contract: take midpoint
        low = float(numbers[0])
        high = float(numbers[1])
        return (low + high) / 2
    elif len(numbers) == 1:
        # Threshold contract
        return float(numbers[0])
    return None

winners["actual_temp"] = winners["subtitle"].apply(extract_temp)

daily_temps = winners[["date", "actual_temp", "last_price", "volume"]].drop_duplicates(subset=["date"])
daily_temps = daily_temps.sort_values("date")

print(daily_temps.head(20))
print(f"\nTotal days: {len(daily_temps)}")

daily_temps.to_csv("kalshi_chicago_daily.csv", index=False)