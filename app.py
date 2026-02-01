from flask import Flask, jsonify, render_template, request
import subprocess
import sys
import os
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

app = Flask(__name__)

KALSHI_TIMEZONE = ZoneInfo("America/New_York")
EVENING_CUTOFF_HOUR = 18  # 6 PM Eastern


def get_prediction_date(date_option, custom_date=None):
    """
    Calculate the prediction date based on the date option.
    Returns a formatted date string.
    """
    eastern_now = datetime.now(KALSHI_TIMEZONE)

    if date_option == "today":
        target_date = eastern_now.date()
    elif date_option == "tomorrow":
        target_date = (eastern_now + timedelta(days=1)).date()
    elif date_option == "custom" and custom_date:
        target_date = datetime.strptime(custom_date, "%Y-%m-%d").date()
    else:  # auto
        hour = eastern_now.hour
        if hour < EVENING_CUTOFF_HOUR:
            target_date = eastern_now.date()
        else:
            target_date = (eastern_now + timedelta(days=1)).date()

    return target_date.strftime("%A, %B %d, %Y")

@app.route('/')
def home():
    return render_template("weather.html")

@app.route('/weather')
def weather():
    return render_template("weather.html")

@app.route('/sports')
def sports():
    return render_template("sports.html")

@app.route("/run-model", methods=["GET"])
def run_model():
    # Get parameters from query string
    kelly_fraction = request.args.get("kelly", "0.50")
    max_bet_fraction = request.args.get("maxbet", "0.20")
    starting_bankroll = request.args.get("bankroll", "40")
    date_option = request.args.get("dateOption", "auto")
    custom_date = request.args.get("customDate", "")
    city_filter = request.args.get("city", "all")

    # Path to ensemble_v9-3-7.py
    script_path = os.path.join(os.path.dirname(__file__), "ensemble_v9-3-7.py")

    # Build command with date arguments
    cmd = [sys.executable, script_path, "--kelly", kelly_fraction, "--maxbet", max_bet_fraction, "--bankroll", starting_bankroll]

    if date_option == "today":
        cmd.append("--today")
    elif date_option == "tomorrow":
        cmd.append("--tomorrow")
    elif date_option == "custom" and custom_date:
        cmd.extend(["--date", custom_date])
    # "auto" uses the script's default behavior (no flag needed)

    # Add city filter if not "all"
    if city_filter and city_filter != "all":
        cmd.extend(["--cities", city_filter])

    # Calculate the prediction date to return in the response
    prediction_date = get_prediction_date(date_option, custom_date)

    # Run the script and capture output with extended timeout (5 minutes for all cities)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for HRRR data fetching across multiple cities
        )

        # Return both raw output and parsed data
        return jsonify({
            "success": True,
            "output": result.stdout,
            "error": result.stderr,
            "raw_output": result.stdout,  # Keep raw output for fallback
            "prediction_date": prediction_date
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "output": "",
            "error": "Model execution timed out after 5 minutes. HRRR data may be slow or unavailable. Try running with fewer cities or wait a few minutes and retry."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "output": "",
            "error": f"Error running model: {str(e)}"
        })

@app.route("/run-sports-model", methods=["GET"])
def run_sports_model():
    # Get parameters from query string
    kelly_fraction = request.args.get("kelly", "0.50")
    starting_bankroll = request.args.get("bankroll", "100")

    # Path to sports_betting_model.py
    script_path = os.path.join(os.path.dirname(__file__), "sports_betting_model.py")

    # Build command
    cmd = [sys.executable, script_path, "--kelly", kelly_fraction, "--bankroll", starting_bankroll]

    # Run the script and capture output with timeout
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        # Return the printed output (stdout)
        return jsonify({
            "success": True,
            "output": result.stdout,
            "error": result.stderr
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "output": "",
            "error": "Model execution timed out after 60 seconds."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "output": "",
            "error": f"Error running model: {str(e)}"
        })

@app.route('/about')
def about():
  return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)