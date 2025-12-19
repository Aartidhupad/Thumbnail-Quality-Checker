from flask import Flask, request, render_template
from predict import predict
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["thumbnail"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            result = predict(filepath)
            return render_template("result.html", prediction=result, image=file.filename)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
