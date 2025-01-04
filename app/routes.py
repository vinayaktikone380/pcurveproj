from flask import Blueprint, request, jsonify
import pickle

bp = Blueprint('main', __name__)

# Load the ML model
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

@bp.route("/")
def home():
    return "Welcome to the Windmill Prediction App!"

@bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    wind_speed = data.get("wind_speed", 0)
    prediction = model.predict([[wind_speed]])
    return jsonify({"predicted_power_output": prediction[0]})
