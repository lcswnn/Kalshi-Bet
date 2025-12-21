from flask import Flask, jsonify, render_template, request
import subprocess
import sys
import os

app = Flask(__name__)

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
    starting_bankroll = request.args.get("bankroll", "40")
    date_option = request.args.get("dateOption", "auto")
    custom_date = request.args.get("customDate", "")
    city_filter = request.args.get("city", "all")

    # Path to ensemble_v11.py
    script_path = os.path.join(os.path.dirname(__file__), "ensemble_v11.py")

    # Build command with date arguments
    cmd = [sys.executable, script_path, "--kelly", kelly_fraction, "--bankroll", starting_bankroll]

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

    # Run the script and capture output with extended timeout (120 seconds)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout for HRRR data fetching
        )
        
        # Return both raw output and parsed data
        return jsonify({
            "success": True,
            "output": result.stdout,
            "error": result.stderr,
            "raw_output": result.stdout  # Keep raw output for fallback
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "output": "",
            "error": "Model execution timed out after 120 seconds. Try running with a single city filter or check if HRRR data is accessible."
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

    # Run the script and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    # Return the printed output (stdout)
    return jsonify({
        "output": result.stdout,
        "error": result.stderr
    })

@app.route('/about')
def about():
  return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)