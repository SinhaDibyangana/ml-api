from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import datetime

app = Flask(__name__)
CORS(app)

# 🔁 Load the new trained model
model = joblib.load("poaching_model.pkl")

# 🗺 Geofence boundaries for each forest
GEOFENCES = {
    "Jim Corbett": {
        "latMin": 29.52, "latMax": 29.535,
        "lngMin": 78.765, "lngMax": 78.780
    },
    "Kanha": {
        "latMin": 22.00, "latMax": 22.20,
        "lngMin": 80.60, "lngMax": 80.80
    },
    "Sundarbans": {
        "latMin": 21.80, "latMax": 21.95,
        "lngMin": 88.85, "lngMax": 89.00
    }
}

@app.route('/')
def home():
    return "Flask ML API is running ✅"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        lat = float(data.get("lat"))
        lng = float(data.get("lng"))
        timestamp = data.get("timestamp")
        forest = data.get("forest", "Jim Corbett")  # default if not passed

        # ⏰ Extract hour and weekday
        dt = datetime.datetime.fromisoformat(timestamp)
        hour = dt.hour
        weekday = dt.weekday()

        # 🧮 Check if outside geofence
        geo = GEOFENCES.get(forest, {})
        outside_fence = int(
            lat < geo.get("latMin", -90) or
            lat > geo.get("latMax", 90) or
            lng < geo.get("lngMin", -180) or
            lng > geo.get("lngMax", 180)
        )

        # 🔢 Prepare input for model
        features = np.array([[lat, lng, hour, weekday, outside_fence]])
        prob = model.predict_proba(features)[0][1]
        risk_label = "High" if prob >= 0.6 else "Low"

        return jsonify({
            "risk_score": round(float(prob), 4),
            "risk_level": risk_label
        })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == '__main__':
    app.run(port=5050, debug=True)
