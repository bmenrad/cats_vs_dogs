from flask import Flask, render_template, request
from PIL import Image
import os
import numpy as np
import requests
import json
import tensorflow as tf

# -------------------------------
# FORCE CPU: Disable all GPU devices
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hide all GPUs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress info/warnings
tf.config.set_visible_devices([], 'GPU')    # hide GPUs at runtime

# -------------------------------
# Flask app setup
# -------------------------------
app = Flask(__name__)

# ✅ Load TensorFlow SavedModel
model_path = "saved_resnet"  # folder containing saved_model.pb + variables/
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0  # normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

# -------------------------------
# Load ImageNet labels
# -------------------------------
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels_json = requests.get(LABELS_URL).text
imagenet_labels = json.loads(labels_json)

# -------------------------------
# Build sets of cat and dog classes
# -------------------------------
imagenet_cats = set()
imagenet_dogs = set()

# List of dog breeds not containing "dog" in the name
extra_dogs = [
    "beagle", "chihuahua", "golden_retriever", "labrador_retriever",
    "pug", "rottweiler", "dalmatian", "siberian_husky"
]

for idx, label_info in imagenet_labels.items():
    name = label_info[1].lower()
    if "cat" in name:
        imagenet_cats.add(name)
    elif "dog" in name or name in extra_dogs:
        imagenet_dogs.add(name)

# -------------------------------
# Prediction function
# -------------------------------
def predict(image_path):
    img_tensor = preprocess_image(image_path)
    output = infer(img_tensor)
    logits = list(output.values())[0]
    pred_idx = int(tf.argmax(logits, axis=1).numpy()[0])
    class_name = imagenet_labels[str(pred_idx)][1].lower()

    if class_name in imagenet_cats:
        return "Cat"
    elif class_name in imagenet_dogs:
        return "Dog"
    else:
        return "Dog"

# -------------------------------
# Flask route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        if "file" not in request.files:
            result = "No file uploaded"
        else:
            file = request.files["file"]
            if file.filename == "":
                result = "No file selected"
            else:
                filepath = os.path.join("static", file.filename)
                file.save(filepath)
                result = predict(filepath)
    return render_template("index.html", result=result)

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
