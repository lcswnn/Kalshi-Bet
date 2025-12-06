from flask import Flask, jsonify, render_template, request
import subprocess
import sys
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/run-model", methods=["GET"])
def run_model():
    # Get parameters from query string
    kelly_fraction = request.args.get("kelly", "0.50")
    starting_bankroll = request.args.get("bankroll", "40")

    # Path to ensemble_v9.py (adjust if needed)
    script_path = os.path.join(os.path.dirname(__file__), "ensemble_v9.py")

    # Run the script and capture output, passing parameters as command-line args
    result = subprocess.run(
        [sys.executable, script_path, "--kelly", kelly_fraction, "--bankroll", starting_bankroll],
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
