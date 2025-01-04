from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)


# Root route
@app.route("/")
def home():
    return render_template("index.html")  # Render an HTML page with a form


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs from the form
        wind_speed = float(request.form["wind_speed"])

        # Make prediction
        prediction = model.predict([[wind_speed]])

        # Return the result
        return render_template(
            "index.html",
            prediction_text=f"Predicted Power Output: {prediction[0]:.2f} kW"
        )
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
