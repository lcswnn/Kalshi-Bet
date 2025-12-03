import requests
import json
import pandas as pd 

#NOAA API Key
NOAA_KEY = "snGAmCkqxGSTXWnipGGsRwRdJvoiFkTM"

#gather historical chicago data
headers = {"token": "snGAmCkqxGSTXWnipGGsRwRdJvoiFkTM"}
params = {
    "datasetid": "GHCND", #Global Historical Climatology Network Daily
    "stationid": "GHCND:USW00014819", #Code for Chicago
    "datatypeid": "TMAX", 
    "startdate": "2020-01-01",
    "enddate": "2020-12-01",
    "units": "standard", 
    "limit": 1000
}

response = requests.get(
    "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
    headers=headers,
    params=params
)
data = response.json()
df = pd.DataFrame(data['results'])
print(df.head())
print(len(df))

