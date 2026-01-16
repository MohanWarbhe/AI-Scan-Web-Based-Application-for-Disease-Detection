from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ===============================
# Upload folder config
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===============================
# Load model
# ===============================
MODEL_PATH = "best_model_accuracy_boost.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ===============================
# Class labels
# ===============================
class_names = {
    0: "Actinic keratosis",
    1: "Atopic Dermatitis",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanocytic nevus",
    5: "Melanoma",
    6: "Squamous cell carcinoma",
    7: "Tinea Ringworm Candidiasis",
    8: "Vascular lesion"
}

# ===============================
# Disease Solutions / Precautions
# ===============================
disease_solutions = {
    "Actinic keratosis": [
        "Avoid excessive sun exposure",
        "Use broad-spectrum sunscreen",
        "Consult a dermatologist"
    ],
    "Atopic Dermatitis": [
        "Moisturize skin regularly",
        "Avoid allergens",
        "Follow prescribed medication"
    ],
    "Benign keratosis": [
        "Usually harmless",
        "Consult doctor if irritation occurs",
        "Removal optional for cosmetic reasons"
    ],
    "Dermatofibroma": [
        "Generally harmless",
        "Avoid scratching",
        "Surgical removal if painful"
    ],
    "Melanocytic nevus": [
        "Monitor size and color changes",
        "Limit sun exposure",
        "Consult dermatologist if needed"
    ],
    "Melanoma": [
        "Seek immediate medical attention",
        "Avoid sun exposure",
        "Early detection saves lives"
    ],
    "Squamous cell carcinoma": [
        "Consult dermatologist promptly",
        "Avoid prolonged sun exposure",
        "Treatment may include surgery"
    ],
    "Tinea Ringworm Candidiasis": [
        "Maintain hygiene",
        "Use antifungal creams",
        "Avoid sharing personal items"
    ],
    "Vascular lesion": [
        "Usually benign",
        "Laser treatment if required",
        "Consult specialist if changes occur"
    ]
}

# ===============================
# Doctor Suggestions (EN + HI)
# ===============================
doctor_suggestions = {
    "Actinic keratosis": {
        "en": "Avoid sun exposure and consult a dermatologist.",
        "hi": "धूप से बचें और त्वचा विशेषज्ञ से परामर्श लें।"
    },
    "Atopic Dermatitis": {
        "en": "Keep skin moisturized and avoid allergens.",
        "hi": "त्वचा को नम रखें और एलर्जी से बचें।"
    },
    "Benign keratosis": {
        "en": "This condition is generally harmless.",
        "hi": "यह स्थिति सामान्यतः हानिरहित होती है।"
    },
    "Dermatofibroma": {
        "en": "Treatment is usually not required.",
        "hi": "आमतौर पर उपचार की आवश्यकता नहीं होती।"
    },
    "Melanocytic nevus": {
        "en": "Observe regularly for any changes.",
        "hi": "किसी भी बदलाव के लिए नियमित निरीक्षण करें।"
    },
    "Melanoma": {
        "en": "Immediate medical attention is required.",
        "hi": "तुरंत चिकित्सकीय सहायता लें।"
    },
    "Squamous cell carcinoma": {
        "en": "Early diagnosis is important.",
        "hi": "शीघ्र निदान अत्यंत आवश्यक है।"
    },
    "Tinea Ringworm Candidiasis": {
        "en": "Use antifungal medication regularly.",
        "hi": "एंटीफंगल दवाओं का नियमित उपयोग करें।"
    },
    "Vascular lesion": {
        "en": "Consult doctor if the lesion changes.",
        "hi": "यदि घाव में बदलाव हो तो डॉक्टर से मिलें।"
    }
}

# ===============================
# MAIN PAGE
# ===============================
@app.route("/", methods=["GET"])
def prediction_page():
    return render_template("prediction.html")

# ===============================
# Prediction Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template("prediction.html")

    file = request.files["image"]
    if file.filename == "":
        return render_template("prediction.html")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Image preprocessing
    img = image.load_img(file_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    preds = model.predict(img_array)
    pred_index = int(np.argmax(preds))
    pred_class = class_names.get(pred_index, "Unknown")

    image_url = url_for("static", filename="uploads/" + filename)

    solutions = disease_solutions.get(pred_class, [])
    doctor = doctor_suggestions.get(pred_class, {"en": "", "hi": ""})

    disclaimer = "This AI prediction is for educational purposes only. Please consult a certified dermatologist."

    return render_template(
        "prediction.html",
        prediction=pred_class,
        image_url=image_url,
        solutions=solutions,
        doctor_en=doctor["en"],
        doctor_hi=doctor["hi"],
        disclaimer=disclaimer
    )

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
