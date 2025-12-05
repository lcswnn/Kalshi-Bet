from flask import Flask, jsonify, render_template
import subprocess
import sys
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/run-model", methods=["GET"])
def run_model():
    # Path to ensemble_v9.py (adjust if needed)
    script_path = os.path.join(os.path.dirname(__file__), "ensemble_v9.py")

    # Run the script and capture output
    result = subprocess.run(
        [sys.executable, script_path],
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
