import requests
from datetime import datetime, timedelta, timezone
import json
import pytz

def reverse_engineer_actual_temp(displayed_fahrenheit):
    """
    Given a displayed Fahrenheit temperature from NWS 5-minute data,
    reverse engineer the possible actual temperature range.
    
    The rounding process:
    1. Actual temp (e.g., 77.6°F) → rounds to whole F° (78°F)
    2. Converts to C° (25.5556°C) → rounds to whole C° (26°C)
    3. Converts back to F° (78.8°F) ← This is what's displayed
    
    We need to work backwards to find the original F° range.
    """
    # The displayed temp came from a rounded Celsius value
    celsius_from_display = (displayed_fahrenheit - 32) * 5/9
    rounded_celsius = round(celsius_from_display)
    
    # Find the range of Celsius values that would round to this
    celsius_min = rounded_celsius - 0.5
    celsius_max = rounded_celsius + 0.5
    
    # Convert back to Fahrenheit to get the range of rounded F° values
    fahrenheit_min = (celsius_min * 9/5) + 32
    fahrenheit_max = (celsius_max * 9/5) + 32
    
    # The "actual high" for purposes of daily summary would be the rounded F°
    # value from the sensor, which is the midpoint of our range
    actual_rounded_f = round((fahrenheit_min + fahrenheit_max) / 2)
    
    return {
        'displayed_f': displayed_fahrenheit,
        'rounded_celsius': rounded_celsius,
        'actual_f_range': (fahrenheit_min - 0.5, fahrenheit_max + 0.5),
        'likely_actual_rounded_f': actual_rounded_f,
        'fahrenheit_range_that_rounds_to_celsius': (fahrenheit_min, fahrenheit_max)
    }

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    if celsius is None:
        return None
    return (celsius * 9/5) + 32

def fetch_nws_observations(station='KMIA', target_date=None):
    """
    Fetch observations from NWS API for a given station.
    
    Args:
        station: 4-letter station identifier (e.g., 'KMIA')
        target_date: datetime object for the date to analyze (defaults to today)
    
    Returns:
        List of observation data
    """
    # NWS API endpoint for observations
    url = f'https://api.weather.gov/stations/{station}/observations'
    
    headers = {
        'User-Agent': '(Weather Data Analysis, contact@example.com)',
        'Accept': 'application/json'
    }
    
    # If no target date provided, use today
    if target_date is None:
        # Get local timezone (Eastern for Miami)
        eastern = pytz.timezone('US/Eastern')
        target_date = datetime.now(eastern).date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    
    try:
        print(f"Fetching data from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'features' not in data:
            print("Unexpected API response format")
            print(json.dumps(data, indent=2)[:500])
            return None
        
        observations = []
        
        # Convert target_date to timezone-aware datetime range
        eastern = pytz.timezone('US/Eastern')
        start_of_day = eastern.localize(datetime.combine(target_date, datetime.min.time()))
        end_of_day = start_of_day + timedelta(days=1)
        
        print(f"Looking for observations between:")
        print(f"  Start: {start_of_day}")
        print(f"  End:   {end_of_day}")
        
        for feature in data['features']:
            props = feature['properties']
            
            # Parse timestamp
            timestamp_str = props.get('timestamp')
            if timestamp_str:
                # Parse ISO format timestamp (already includes timezone info)
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                # Convert to Eastern time for comparison
                timestamp_eastern = timestamp.astimezone(eastern)
                
                # Only include observations from the target date
                if timestamp_eastern < start_of_day or timestamp_eastern >= end_of_day:
                    continue
            else:
                continue
            
            # Get temperature (comes in Celsius from API)
            temp_c = props.get('temperature', {}).get('value')
            
            if temp_c is not None:
                temp_f = celsius_to_fahrenheit(temp_c)
                
                observations.append({
                    'timestamp': timestamp_str,
                    'datetime': timestamp,
                    'datetime_eastern': timestamp_eastern,
                    'temp_celsius': temp_c,
                    'temp_fahrenheit': temp_f,
                    'text_description': props.get('textDescription', 'N/A'),
                    'raw_message': props.get('rawMessage', '')
                })
        
        # Sort by time
        observations.sort(key=lambda x: x['datetime'])
        
        return observations
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_displayed_temperature(temp_celsius):
    """
    Simulate how NWS displays temperature after rounding.
    C° → round → F° conversion
    """
    rounded_celsius = round(temp_celsius)
    displayed_fahrenheit = celsius_to_fahrenheit(rounded_celsius)
    return displayed_fahrenheit

def analyze_observations(observations):
    """
    Analyze observations to find actual vs displayed highs.
    """
    if not observations:
        return None
    
    # Sort by temperature
    observations_with_analysis = []
    
    for obs in observations:
        temp_c = obs['temp_celsius']
        temp_f = obs['temp_fahrenheit']
        
        # This is what NWS would display (after rounding C° and converting)
        displayed_f = get_displayed_temperature(temp_c)
        
        # This is what the actual reading would be (rounded to whole F°)
        actual_rounded_f = round(temp_f)
        
        # Reverse engineer to verify
        reverse_analysis = reverse_engineer_actual_temp(displayed_f)
        
        observations_with_analysis.append({
            **obs,
            'displayed_fahrenheit': displayed_f,
            'actual_rounded_fahrenheit': actual_rounded_f,
            'reverse_analysis': reverse_analysis
        })
    
    # Find max by displayed temperature
    max_displayed = max(observations_with_analysis, key=lambda x: x['displayed_fahrenheit'])
    
    # Find max by actual temperature
    max_actual = max(observations_with_analysis, key=lambda x: x['actual_rounded_fahrenheit'])
    
    return {
        'observations': observations_with_analysis,
        'max_displayed': max_displayed,
        'max_actual': max_actual
    }

def main():
    station = 'KMIA'  # Miami International Airport
    
    # Get today's date in Eastern time
    eastern = pytz.timezone('US/Eastern')
    target_date = datetime.now(eastern).date()
    
    print(f"Analyzing temperature data for {station} on {target_date}")
    print("="*60)
    
    observations = fetch_nws_observations(station, target_date)
    
    if not observations:
        print("No observations found for the specified date.")
        print("\nNote: The NWS API typically only has data for the last 24-48 hours.")
        print("If you're looking for future data, it won't be available yet.")
        return
    
    print(f"\nFound {len(observations)} observations for {target_date}")
    
    if len(observations) == 0:
        print("\nNo observations available for this date yet.")
        return
    
    # Analyze the data
    result = analyze_observations(observations)
    
    if not result:
        print("Failed to analyze observations")
        return
    
    print("\n" + "="*60)
    print(f"TEMPERATURE ANALYSIS FOR {station} - {target_date}")
    print("="*60)
    
    max_displayed = result['max_displayed']
    max_actual = result['max_actual']
    
    print(f"\nMAXIMUM DISPLAYED TEMPERATURE (what you see on weather apps):")
    print(f"  Time: {max_displayed['datetime_eastern'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Raw sensor: {max_displayed['temp_celsius']:.2f}°C ({max_displayed['temp_fahrenheit']:.2f}°F)")
    print(f"  Displayed as: {max_displayed['displayed_fahrenheit']:.1f}°F")
    print(f"  (from rounded {max_displayed['reverse_analysis']['rounded_celsius']}°C)")
    
    print(f"\nACTUAL DAILY HIGH (what will be in official summary):")
    print(f"  Time: {max_actual['datetime_eastern'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Actual rounded: {max_actual['actual_rounded_fahrenheit']}°F")
    
    if max_displayed['actual_rounded_fahrenheit'] != max_actual['actual_rounded_fahrenheit']:
        print(f"\n⚠️  DISCREPANCY DETECTED!")
        print(f"  The highest displayed temperature ({max_displayed['displayed_fahrenheit']:.1f}°F)")
        print(f"  corresponds to an actual {max_displayed['actual_rounded_fahrenheit']}°F,")
        print(f"  but the actual daily high is {max_actual['actual_rounded_fahrenheit']}°F")
    else:
        print(f"\n✓ No discrepancy - displayed and actual highs match at {max_actual['actual_rounded_fahrenheit']}°F")
    
    # Show all observations
    print("\n" + "="*60)
    print(f"ALL OBSERVATIONS FOR {target_date}:")
    print("="*60)
    print(f"{'Time (ET)':<20} {'Raw (°C)':<12} {'Raw (°F)':<12} {'Displayed':<12} {'Actual':<10}")
    print("-"*70)
    
    for obs in result['observations']:
        time_str = obs['datetime_eastern'].strftime('%m/%d %H:%M')
        print(f"{time_str:<20} {obs['temp_celsius']:>6.2f}°C    {obs['temp_fahrenheit']:>6.2f}°F    "
              f"{obs['displayed_fahrenheit']:>6.1f}°F      {obs['actual_rounded_fahrenheit']:>3.0f}°F")
    
    print("\n" + "="*60)
    print("EXPLANATION:")
    print("="*60)
    print("- 'Raw' = Original sensor reading from NWS API")
    print("- 'Displayed' = What appears on weather apps (C° rounded → F°)")
    print("- 'Actual' = What will appear in daily summary (raw F° rounded)")
    print("\nThe rounding from C° to F° can inflate displayed temperatures!")
    
    # Export to JSON
    filename = f'kmia_temperature_analysis_{target_date}.json'
    print("\n" + "="*60)
    print(f"Exporting data to '{filename}'...")
    export_data = {
        'station': station,
        'date': str(target_date),
        'analysis_time': datetime.now(timezone.utc).isoformat(),
        'max_displayed': {
            'time': max_displayed['datetime'].isoformat(),
            'time_eastern': max_displayed['datetime_eastern'].isoformat(),
            'raw_celsius': max_displayed['temp_celsius'],
            'raw_fahrenheit': max_displayed['temp_fahrenheit'],
            'displayed_fahrenheit': max_displayed['displayed_fahrenheit'],
            'actual_rounded_fahrenheit': max_displayed['actual_rounded_fahrenheit']
        },
        'max_actual': {
            'time': max_actual['datetime'].isoformat(),
            'time_eastern': max_actual['datetime_eastern'].isoformat(),
            'raw_celsius': max_actual['temp_celsius'],
            'raw_fahrenheit': max_actual['temp_fahrenheit'],
            'displayed_fahrenheit': max_actual['displayed_fahrenheit'],
            'actual_rounded_fahrenheit': max_actual['actual_rounded_fahrenheit']
        },
        'all_observations': [
            {
                'time': obs['datetime'].isoformat(),
                'time_eastern': obs['datetime_eastern'].isoformat(),
                'raw_celsius': obs['temp_celsius'],
                'raw_fahrenheit': obs['temp_fahrenheit'],
                'displayed_fahrenheit': obs['displayed_fahrenheit'],
                'actual_rounded_fahrenheit': obs['actual_rounded_fahrenheit']
            }
            for obs in result['observations']
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("Data exported successfully!")

if __name__ == "__main__":
    main()