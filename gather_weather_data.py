import requests
import csv
import time
from datetime import date

NOAA_KEY = "snGAmCkqxGSTXWnipGGsRwRdJvoiFkTM"
headers = {"token": NOAA_KEY}

# Define cities with their station IDs and output files
cities = [
    {
        "name": "Chicago",
        "station_id": "GHCND:USW00014819",  # Chicago Midway
        "csv_file": "weather_data_chicago.csv"
    },
    {
        "name": "New York City",
        "station_id": "GHCND:USW00094728",  # NYC Central Park
        "csv_file": "weather_data_nyc.csv"
    }
]

def fetch_weather_data(city):
    """Fetch historical weather data for a city, going back year by year."""
    print(f"\n{'='*50}")
    print(f"Fetching data for {city['name']}...")
    print(f"{'='*50}")
    
    today = date.today()
    year = today.year
    end_year = 1997  # How far back to go
    
    csv_file = city["csv_file"]
    is_first_batch = True
    total_records = 0
    
    while year >= end_year:
        # Set date range for this year
        start_date = f"{year}-01-01"
        if year == today.year:
            end_date = today.strftime("%Y-%m-%d")
        else:
            end_date = f"{year}-12-31"
        
        params = {
            "datasetid": "GHCND",
            "stationid": city["station_id"],
            "datatypeid": "TMAX",
            "startdate": start_date,
            "enddate": end_date,
            "units": "standard",
            "limit": 1000
        }
        
        response = requests.get(
            "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            print(f"  Year {year}: Error {response.status_code}")
            year -= 1
            time.sleep(0.3)
            continue
        
        data = response.json()
        
        if "results" not in data or len(data["results"]) == 0:
            print(f"  Year {year}: No data")
            year -= 1
            time.sleep(0.3)
            continue
        
        records = data["results"]
        
        # Write to CSV
        mode = 'w' if is_first_batch else 'a'
        with open(csv_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            if is_first_batch:
                writer.writeheader()
                is_first_batch = False
            writer.writerows(records)
        
        total_records += len(records)
        print(f"  Year {year}: {len(records)} records (total: {total_records})")
        
        year -= 1
        time.sleep(0.3)  # Rate limiting
    
    print(f"\n✅ {city['name']}: {total_records} total records saved to {csv_file}")
    return total_records


# Fetch data for all cities
if __name__ == "__main__":
    print("NOAA Weather Data Fetcher")
    print("=" * 50)
    
    for city in cities:
        fetch_weather_data(city)
        time.sleep(1)  # Pause between cities
    
    print("\n" + "=" * 50)
    print("COMPLETE!")
    print("=" * 50)