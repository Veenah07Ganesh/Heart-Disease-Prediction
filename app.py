from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

# Load model and scaler
model = pickle.load(open("XGB_BRFSSfinal.pkl", "rb"))
scaler = joblib.load("scaler.pkl")

# Load feature names
with open("feature_names.txt") as f:
    feature_names = [line.strip() for line in f]

# Age mapping function
def map_age_to_category(age):
    age = int(age)
    if 18 <= age <= 24: return 0
    elif 25 <= age <= 29: return 1
    elif 30 <= age <= 34: return 2
    elif 35 <= age <= 39: return 3
    elif 40 <= age <= 44: return 4
    elif 45 <= age <= 49: return 5
    elif 50 <= age <= 54: return 6
    elif 55 <= age <= 59: return 7
    elif 60 <= age <= 64: return 8
    elif 65 <= age <= 69: return 9
    elif 70 <= age <= 74: return 10
    elif 75 <= age <= 79: return 11
    else: return 12

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html", prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_age = request.form.get("RawAge")
        age_category = map_age_to_category(raw_age)

        input_data = []
        for f in feature_names:
            if f == "AgeCategory":
                input_data.append(age_category)
            else:
                val = request.form.get(f)
                input_data.append(float(val) if val is not None else 0)

        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]

        result = "⚠️ High Risk of Heart Disease" if prediction == 1 else "✅ Low Risk of Heart Disease"
        return render_template("prediction.html", prediction_text=result)

    except Exception as e:
        return render_template("prediction.html", prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
