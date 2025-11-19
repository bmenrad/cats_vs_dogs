from flask import Flask, render_template, request
from PIL import Image
import os
import numpy as np
import requests
import json
import tensorflow as tf
import cv2

# -------------------------------
# FORCE CPU
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], 'GPU')

# -------------------------------
# Flask app setup
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load TensorFlow SavedModel via TFSMLayer (Keras 3)
# -------------------------------
from keras.layers import TFSMLayer
saved_model_path = "saved_resnet"
model = tf.keras.Sequential([
    TFSMLayer(saved_model_path, call_endpoint="serving_default")
])
# Hinweis: inference-only

# -------------------------------
# ImageNet labels
# -------------------------------
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels_json = requests.get(LABELS_URL).text
imagenet_labels = json.loads(labels_json)

imagenet_cats = set()
imagenet_dogs = set()
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
# Image preprocessing
# -------------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

# -------------------------------
# Grad-CAM (für TFSMLayer)
# -------------------------------
def make_gradcam_heatmap_tfsmlayer(img_tensor, tfsmlayer_model):
    """
    Berechnet Grad-CAM auf dem TFSMLayer-Modell.
    """
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds_dict = tfsmlayer_model(img_tensor)  # dict zurück
        logits = preds_dict['logits']             # Tensor extrahieren
        pred_index = tf.argmax(logits, axis=-1)[0]  # Batch-Index
        class_channel = logits[:, pred_index]

    grads = tape.gradient(class_channel, img_tensor)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(img_tensor, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (224, 224))
    return tf.squeeze(heatmap).numpy(), int(pred_index.numpy())

def save_and_overlay_heatmap(img_path, heatmap, cam_path="static/gradcam.png", alpha=0.4):
    """
    Legt Heatmap über Originalbild und speichert das Ergebnis.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img
    cv2.imwrite(cam_path, cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_RGB2BGR))
    return cam_path

# -------------------------------
# Prediction mit Grad-CAM
# -------------------------------
def predict_with_gradcam(image_path):
    img_tensor = preprocess_image(image_path)
    heatmap, pred_idx = make_gradcam_heatmap_tfsmlayer(img_tensor, model)
    cam_path = save_and_overlay_heatmap(image_path, heatmap)

    class_name = imagenet_labels[str(pred_idx)][1].lower()
    if class_name in imagenet_cats:
        label = "Cat"
    elif class_name in imagenet_dogs:
        label = "Dog"
    else:
        label = "Dog"

    return label, cam_path

# -------------------------------
# Flask route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    cam_path = ""
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
                result, cam_path = predict_with_gradcam(filepath)
    return render_template("index.html", result=result, cam_path=cam_path)

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
