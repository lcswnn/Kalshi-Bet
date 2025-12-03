import requests
import json
import csv

#gather historical chicago data
results = True
year = 2024
csv_file = "weather_data.csv"
is_first_batch = True

while results == True:
  NOAA_KEY = "snGAmCkqxGSTXWnipGGsRwRdJvoiFkTM"
  headers = {"token": NOAA_KEY}
  params = {
      "datasetid": "GHCND", #Global Historical Climatology Network Daily
      "stationid": "GHCND:USW00014819", #Code for Chicago
      "datatypeid": "TMAX",
      "startdate": f"{year}-01-01",
      "enddate": f"{year}-12-02",
      "units": "standard",
      "limit": 1000
  }

  response = requests.get(
      "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
      headers=headers,
      params=params
  )

  print(f"Status Code: {response.status_code}")
  if response.status_code != 200:
      print(f"Error Response: {response.text}")
      results = False
      break
  else:
      data = response.json()
      records = data['results']

      if records:
          mode = 'w' if is_first_batch else 'a'
          with open(csv_file, mode, newline='') as f:
              writer = csv.DictWriter(f, fieldnames=records[0].keys())
              if is_first_batch:
                  writer.writeheader()
                  is_first_batch = False
              writer.writerows(records)

          print(f"Wrote {len(records)} records for year {year}")

      year = year - 1

