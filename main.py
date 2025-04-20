from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")
print("✅ Model loaded successfully.")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    if request.method == "POST":
        review = request.form["review"]
        probas = model.predict_proba([review])[0]
        pred_class = np.argmax(probas)
        prediction = "Positive" if pred_class == 1 else "Negative"
        confidence = round(probas[pred_class] * 100, 2)
    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5678, debug=True)
