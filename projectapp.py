from flask import Flask, request, jsonify
from model import get_prediction

app = Flask(__name__)


@app.route("/predict-digit", methods = ["POST"])
def predict():
    image = request.files.get("digit")
    pred = get_prediction(image)
    return jsonify({
        "prediction":pred
    })

if( __name__ == "__main__"):
    app.run(debug=True)