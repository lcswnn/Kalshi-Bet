import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import io
import numpy as np

def fetch_observations(station_id, start_date, end_date):
    url = (
        "https://www.ncei.noaa.gov/access/services/data/v1"
        "?dataset=daily-summaries"
        "&dataTypes=TMIN,TMAX"
        f"&stations={station_id}"
        f"&startDate={start_date}"
        f"&endDate={end_date}"
        "&includeAttributes=false"
        "&format=csv"
        "&units=standard"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch observations: {response.text}")
    df = pd.read_csv(io.StringIO(response.text))
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    return df

def fetch_mos_forecasts(station, start_utc, end_utc):
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
        f"?station={station}"
        "&model=GFS"  # Changed from MAV → GFS for GFS MOS
        f"&sts={start_utc.strftime('%Y-%m-%dT%H:%M')}Z"
        f"&ets={end_utc.strftime('%Y-%m-%dT%H:%M')}Z"
        "&format=csv"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch MOS: {response.status_code} - {response.text}")
    
    # Also add a check if the content is actually CSV data or an error message
    if "error" in response.text.lower() or len(response.text.strip()) < 100:
        raise ValueError(f"MOS response looks invalid: {response.text[:200]}...")
    
    df = pd.read_csv(io.StringIO(response.text), comment='#')
    df['runtime'] = pd.to_datetime(df['runtime'])
    df['ftime'] = pd.to_datetime(df['ftime'])
    return df

def get_local_day(utc_time, tz):
    return utc_time.astimezone(tz).date()

def analyze_accuracy(station, station_id, tz_name, start_date, end_date):
    tz = ZoneInfo(tz_name)
    
    # Fetch observations
    obs = fetch_observations(station_id, start_date, end_date)
    print(f"Fetched {len(obs)} observation days for {station} "
          f"({obs.index.min().date()} → {obs.index.max().date()})")
    
    # Fetch MOS a bit earlier/later to be safe
    start_utc = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc) - timedelta(days=3)
    end_utc = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) + timedelta(days=2)
    mos = fetch_mos_forecasts(station, start_utc, end_utc)
    print(f"Fetched {len(mos)} MOS rows for {station} "
          f"(runtimes from {mos['runtime'].min()} to {mos['runtime'].max()})")
    
    results = []
    
    for date_str in pd.date_range(start_date, end_date, freq='D'):
        date = date_str.date()
        
        # Actual high
        if date_str not in obs.index:
            print(f"No observation for {date}")
            continue
        actual_high = obs.loc[date_str, 'TMAX']
        if pd.isna(actual_high):
            print(f"Missing TMAX for {date}")
            continue
        
        # Local day start in UTC (midnight local → UTC)
        local_start = datetime.combine(date, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
        
        # Get ALL runs before the day starts, then take the latest one
        available_runs = mos[mos['runtime'] < local_start]
        if available_runs.empty:
            print(f"No MOS runs available before {date} for {station}")
            continue
        
        latest_runtime = available_runs['runtime'].max()
        run_mos = mos[mos['runtime'] == latest_runtime]
        
        print(f"Using MOS run from {latest_runtime.date()} {latest_runtime.time()} "
              f"for forecast valid on {date}")
        
        # Forecast window: widen slightly to catch boundary issues (±3 hours)
        window_start = local_start - timedelta(hours=3)
        window_end   = local_start + timedelta(days=1) + timedelta(hours=3)
        
        day_forecasts = run_mos[(run_mos['ftime'] >= window_start) & (run_mos['ftime'] <= window_end)]
        
        if day_forecasts.empty:
            print(f"No hourly forecasts found in window for {date} (run {latest_runtime.date()})")
            continue
        
        forecasted_temps = day_forecasts['tmp'].values
        forecasted_high = np.max(forecasted_temps) if len(forecasted_temps) > 0 else np.nan
        
        if np.isnan(forecasted_high):
            print(f"No valid temps in forecast for {date}")
            continue
        
        difference = forecasted_high - actual_high
        
        results.append({
            'date': date,
            'actual_high': actual_high,
            'forecasted_high': forecasted_high,
            'difference': difference
        })
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print(f"\nNo valid forecast-obs pairs found for {station} in {start_date} to {end_date}.")
        return None
    
    # Summary stats
    mae = np.mean(np.abs(df_results['difference']))
    bias = np.mean(df_results['difference'])
    within_2 = np.mean(np.abs(df_results['difference']) <= 2) * 100
    within_5 = np.mean(np.abs(df_results['difference']) <= 5) * 100
    
    print(f"\nSummary for {station}:")
    print(f"Days compared: {len(df_results)}")
    print(f"Mean Absolute Error: {mae:.2f} °F")
    print(f"Bias (forecast - actual): {bias:.2f} °F")
    print(f"Percentage within 2°F: {within_2:.2f}%")
    print(f"Percentage within 5°F: {within_5:.2f}%")
    print("\nDaily results:")
    print(df_results.to_string(index=False))
    
    return df_results

# Parameters
start_date = '2025-06-01'  # Adjust as needed
end_date = '2026-01-31'    # Adjust as needed

# For KMIA
print("Analyzing KMIA (Miami)")
kmia_results = analyze_accuracy('KMIA', 'GHCND:USW00012839', 'America/New_York', start_date, end_date)
if kmia_results is not None:
    print(kmia_results)

# For KAUS
print("\nAnalyzing KAUS (Austin)")
kaus_results = analyze_accuracy('KAUS', 'GHCND:USW00013904', 'America/Chicago', start_date, end_date)
if kaus_results is not None:
    print(kaus_results)