import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from utils.predict import predict_disease
from utils.remedy_fetcher import fetch_remedies_offline
from utils.report_generator import generate_pdf_report

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

results_storage = []

def safe_path(path):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    return abs_path

@app.route("/", methods=["GET", "POST"])
def index():
    global results_storage
    if request.method == "POST":
        files = request.files.getlist("file")
        results_storage = []

        for idx, file in enumerate(files):
            if file:
                filename = f"image_{idx}.jpg"
                img_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(img_path)

                gradcam_path = os.path.join(UPLOAD_FOLDER, f"grad_{filename}")
                pred_class, confidence, gradcam_path = predict_disease(img_path, gradcam_path)

                remedy = fetch_remedies_offline(pred_class)
                plant_name, disease_name = pred_class.split("___")
                plant_name = plant_name.replace("_", " ")
                disease_name = disease_name.replace("_", " ")

                results_storage.append({
                    "image_path": img_path,
                    "gradcam_path": gradcam_path,
                    "plant": plant_name,
                    "disease": disease_name,
                    "confidence": confidence,
                    "cause": remedy["cause"],
                    "natural": remedy["natural"],
                    "pesticide": remedy["pesticide"]
                })

        return redirect(url_for("show_results"))

    return render_template("index.html")

@app.route("/results")
def show_results():
    return render_template("results_index.html", results=results_storage)

@app.route("/result/<int:idx>")
def single_result(idx):
    if 0 <= idx < len(results_storage):
        return render_template("result_single.html", r=results_storage[idx], idx=idx, total=len(results_storage))
    return redirect(url_for("index"))

@app.route("/download/<int:idx>")
def download_report(idx):
    if 0 <= idx < len(results_storage):
        r = results_storage[idx]
        pdf_path = os.path.join(UPLOAD_FOLDER, f"report_{idx}.pdf")

        image_path = safe_path(r["image_path"])
        gradcam_path = safe_path(r["gradcam_path"])

        generate_pdf_report(
            pdf_path,
            image_path,
            gradcam_path,
            r["plant"],
            r["disease"],
            r["confidence"],
            r["cause"],
            r["natural"],
            r["pesticide"]
        )
        return send_file(pdf_path, as_attachment=True)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
"""

# Save this to a file so user can download and use it directly
app_path = "/mnt/data/app_updated.py"
with open(app_path, "w") as f:
    f.write(app_py_code)

app_path

Analyzed
python
Always show details

Copy
# Recreate the updated app.py file after code execution environment reset

app_py_code = """
import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from utils.predict1 import predict_disease
from utils.remedy_fetcher import fetch_remedies_offline
from utils.report_generator import generate_pdf_report

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

results_storage = []

def safe_path(path):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    return abs_path

@app.route("/", methods=["GET", "POST"])
def index():
    global results_storage
    if request.method == "POST":
        files = request.files.getlist("file")
        results_storage = []

        for idx, file in enumerate(files):
            if file:
                filename = f"image_{idx}.jpg"
                img_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(img_path)

                gradcam_path = os.path.join(UPLOAD_FOLDER, f"grad_{filename}")
                pred_class, confidence, gradcam_path = predict_disease(img_path, gradcam_path)

                remedy = fetch_remedies_offline(pred_class)
                plant_name, disease_name = pred_class.split("___")
                plant_name = plant_name.replace("_", " ")
                disease_name = disease_name.replace("_", " ")

                results_storage.append({
                    "image_path": img_path,
                    "gradcam_path": gradcam_path,
                    "plant": plant_name,
                    "disease": disease_name,
                    "confidence": confidence,
                    "cause": remedy["cause"],
                    "natural": remedy["natural"],
                    "pesticide": remedy["pesticide"]
                })

        return redirect(url_for("show_results"))

    return render_template("index.html")

@app.route("/results")
def show_results():
    return render_template("results_index.html", results=results_storage)

@app.route("/result/<int:idx>")
def single_result(idx):
    if 0 <= idx < len(results_storage):
        return render_template("result_single.html", r=results_storage[idx], idx=idx, total=len(results_storage))
    return redirect(url_for("index"))

@app.route("/download/<int:idx>")
def download_report(idx):
    if 0 <= idx < len(results_storage):
        r = results_storage[idx]
        pdf_path = os.path.join(UPLOAD_FOLDER, f"report_{idx}.pdf")

        image_path = safe_path(r["image_path"])
        gradcam_path = safe_path(r["gradcam_path"])

        generate_pdf_report(
            pdf_path,
            image_path,
            gradcam_path,
            r["plant"],
            r["disease"],
            r["confidence"],
            r["cause"],
            r["natural"],
            r["pesticide"]
        )
        return send_file(pdf_path, as_attachment=True)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
