from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import re
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "ACDIRP_Aviation_Cascading_Risk_Web.joblib"
)

model = joblib.load(MODEL_PATH)
print("MODEL LOADED SUCCESSFULLY")


# ---------------------------------------------------------
# Feature engineering functions
# ---------------------------------------------------------

def contains_any(text, words):
    text = str(text).lower()
    return any(word in text for word in words)


fire_words = [
    "fire", "burn", "burning", "flame", "flames"
]

explosion_words = [
    "explosion", "explode", "exploded", "blast"
]

infrastructure_words = [
    "building", "house", "road", "bridge",
    "railway", "railroad", "power line",
    "powerline", "structure"
]

hazard_words = [
    "fuel", "spill", "chemical",
    "hazmat", "hazardous"
]


def create_features(form_data):

    aboard = float(form_data.get("Aboard", 0) or 0)
    ground = float(form_data.get("Ground", 0) or 0)

    operator = str(form_data.get("Operator", "") or "")
    aircraft_type = str(form_data.get("Type", "") or "")
    location = str(form_data.get("Location", "") or "")
    route = str(form_data.get("Route", "") or "")
    time_value = str(form_data.get("Time", "") or "")

    # Combine text fields for keyword detection
    text = " ".join([
        operator,
        aircraft_type,
        location,
        route,
        time_value
    ]).lower()

    # -----------------------------------------------------
    # Basic engineered numerical features
    # -----------------------------------------------------

    survival_count = max(aboard - 0, 0)

    fatality_count = 0

    fatality_count_log = np.log1p(fatality_count)

    aboard_log = np.log1p(aboard)

    # -----------------------------------------------------
    # Date-related features
    # -----------------------------------------------------

    accident_date = str(form_data.get("AccidentDate", "") or "")

    accident_datetime = pd.to_datetime(
        accident_date,
        errors="coerce"
    )

    if pd.notna(accident_datetime):
        accident_year = int(accident_datetime.year)
        accident_month = int(accident_datetime.month)
        accident_dayofweek = int(accident_datetime.dayofweek)
    else:
        accident_year = 2026
        accident_month = 10
        accident_dayofweek = 4

    # -----------------------------------------------------
    # Secondary-impact indicators
    # -----------------------------------------------------

    secondary_indicator = "1" if (
        ground > 0
        or contains_any(text, fire_words)
        or contains_any(text, explosion_words)
        or contains_any(text, infrastructure_words)
        or contains_any(text, hazard_words)
    ) else "0"

    ground_impact_flag = str(ground > 0)

    # -----------------------------------------------------
    # Flight-size classification
    # -----------------------------------------------------

    if aboard >= 200:
        large_flight_flag = "Very_Large"
    elif aboard >= 100:
        large_flight_flag = "Large"
    elif aboard >= 50:
        large_flight_flag = "Medium"
    else:
        large_flight_flag = "Small"

    # -----------------------------------------------------
    # Accident scale
    # -----------------------------------------------------

    if aboard >= 200 or ground >= 20:
        accident_scale = "Very_Large"
    elif aboard >= 100 or ground >= 10:
        accident_scale = "Large"
    elif aboard >= 50 or ground >= 5:
        accident_scale = "Medium"
    else:
        accident_scale = "Small"

    # -----------------------------------------------------
    # Text indicators
    # -----------------------------------------------------

    fire_indicator = contains_any(text, fire_words)

    explosion_indicator = contains_any(
        text,
        explosion_words
    )

    infrastructure_indicator = contains_any(
        text,
        infrastructure_words
    )

    hazard_indicator = contains_any(
        text,
        hazard_words
    )

    # -----------------------------------------------------
    # Create model input
    # -----------------------------------------------------

    input_data = pd.DataFrame([{
        "accident_year": accident_year,
        "accident_month": accident_month,
        "accident_dayofweek": accident_dayofweek,

        "Aboard": aboard,
        "Ground": ground,

        "survival_count": survival_count,
        "fatality_count_log": fatality_count_log,
        "aboard_log": aboard_log,

        "Operator": operator,
        "Type": aircraft_type,
        "Location": location,
        "Route": route,
        "Time": time_value,

        "secondary_indicator": secondary_indicator,
        "ground_impact_flag": ground_impact_flag,
        "large_flight_flag": large_flight_flag,
        "accident_scale": accident_scale,

        "fire_indicator": str(fire_indicator),
        "explosion_indicator": str(explosion_indicator),
        "infrastructure_indicator": str(infrastructure_indicator),
        "hazard_indicator": str(hazard_indicator)
    }])

    return input_data


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("prediction.html")

    try:

        input_data = create_features(request.form)

        print("\nMODEL INPUT:")
        print(input_data.to_string())

        prediction = model.predict(input_data)[0]

        probability_text = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = model.classes_

            probability_text = {
                str(cls): round(float(prob) * 100, 2)
                for cls, prob in zip(classes, probabilities)
            }

        return render_template(
            "result.html",
            prediction=prediction,
            probabilities=probability_text
        )

    except Exception as e:

        return f"""
        <h1>Prediction Error</h1>
        <pre>{e}</pre>
        <br>
        <a href="/">Go Back</a>
        """, 500


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "ACDIRP Flask application is running"
    }




# ---------------------------------------------------------
# Methodology page
# ---------------------------------------------------------

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


# ---------------------------------------------------------
# About page
# ---------------------------------------------------------

@app.route("/about")
def about():
    return render_template("about.html")


# ---------------------------------------------------------
# Run application
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run()


