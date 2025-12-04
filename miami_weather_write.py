import requests
import csv
import time
from datetime import date

NOAA_KEY = "snGAmCkqxGSTXWnipGGsRwRdJvoiFkTM"
headers = {"token": NOAA_KEY}

city_name = "Miami"
station_id = "GHCND:USW00012839"  # Miami International Airport
csv_file = "weather_data_miami.csv"

print(f"Fetching data for {city_name}...")

today = date.today()
year = today.year
end_year = 1997

is_first_batch = True
total_records = 0

while year >= end_year:
    start_date = f"{year}-01-01"
    if year == today.year:
        end_date = today.strftime("%Y-%m-%d")
    else:
        end_date = f"{year}-12-31"
    
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "datatypeid": "TMAX",
        "startdate": start_date,
        "enddate": end_date,
        "units": "standard",
        "limit": 1000
    }
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.get(
            "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            print(f"  Rate limited, waiting 10 seconds...")
            time.sleep(10)
        else:
            print(f"  Attempt {attempt + 1}: Status {response.status_code}")
            time.sleep(3)
    
    print(f"Year {year}: Status {response.status_code}")
    
    if response.status_code != 200:
        print(f"  Error: {response.text}")
        print(f"  Waiting 5 seconds before continuing...")
        time.sleep(5)
        year -= 1
        continue
    
    data = response.json()
    
    if "results" not in data or len(data["results"]) == 0:
        print(f"  No data for {year}")
        year -= 1
        time.sleep(2)
        continue
    
    records = data["results"]
    
    mode = 'w' if is_first_batch else 'a'
    with open(csv_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        if is_first_batch:
            writer.writeheader()
            is_first_batch = False
        writer.writerows(records)
    
    total_records += len(records)
    print(f"  Wrote {len(records)} records (total: {total_records})")
    
    year -= 1
    time.sleep(1.3)

print(f"\n✅ {city_name}: {total_records} records saved to {csv_file}")