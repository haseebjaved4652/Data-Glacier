import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set your own secret key

# Load the machine learning model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    try:
        float_features = [float(request.form.get(name)) for name in feature_names]
        features = [np.array(float_features)]

        prediction = model.predict(features)

        if prediction[0] == 0:
            prediction_text = "No, the person does not have diabetes."
        else:
            prediction_text = "Yes, the person has diabetes."

        return render_template("index.html", prediction_text=prediction_text)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template("index.html", prediction_text=error_message)


if __name__ == "__main__":
    app.run(debug=True)
