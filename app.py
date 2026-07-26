from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load pretrained emotion detection model
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    if text == "":
        return render_template(
            "index.html",
            emotion="Please enter some text."
        )

    result = classifier(text)

    emotion = result[0][0]["label"]
    score = result[0][0]["score"]

    return render_template(
        "index.html",
        emotion=emotion,
        confidence=round(score * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
