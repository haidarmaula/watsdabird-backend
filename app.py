from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras_cv_attention_models import *
from tensorflow.keras.models import load_model
import tensorflow as tf
from audio_util import AudioUtil

# ----------------------------
# Configuration: paths, allowed file types, model details, and audio parameters.
# ----------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
MODEL_PATH = 'model.keras'
TARGET_SHAPE = (64, 108)
NUM_CLASSES = 4
SAMPLE_RATE = 22050
WINDOW_MS = 5000
OVERLAP_MS = 2500

CLASS_MAP = {
    0: "AtlanticCanary",
    1: "Sooty-headedBulbul",
    2: "ZebraDove",
    3: "MoustachedBabbler",
}

# ----------------------------
# Initialize Flask app and load the trained bird sound classification model.
# Ensure the upload directory exists.
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# ----------------------------
# Check if the file has a valid audio extension.
# ----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------------------
# Convert an audio segment into a mel-spectrogram, resize and normalize it
# so it can be fed into the neural network model.
# ----------------------------
def preprocess_segment(segment_audio):
    spec = AudioUtil.melspectrogram(segment_audio)
    spec = tf.image.resize(spec[..., np.newaxis], TARGET_SHAPE).numpy()
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec

# ----------------------------
# Predict the bird species from the uploaded audio by:
# - Splitting the audio into overlapping segments
# - Converting each segment into a spectrogram
# - Averaging the prediction probabilities over all segments
# ----------------------------
def predict_file(audio_path):
    audio = AudioUtil.open(audio_path, sample_rate=SAMPLE_RATE, mono=True)
    segments = AudioUtil.split(audio, WINDOW_MS, OVERLAP_MS)
    if not segments:
        raise ValueError("No valid audio segments found.")

    specs = [preprocess_segment(seg) for seg in segments]
    batch = np.stack(specs, axis=0)

    preds = model.predict(batch, verbose=0)
    avg_probs = preds.mean(axis=0)
    pred_idx = int(np.argmax(avg_probs))
    return avg_probs.tolist(), pred_idx

# ----------------------------
# Root endpoint for testing if the server is running.
# ----------------------------
@app.route('/')
def hello():
    return 'Hello, World!', 200

# ----------------------------
# Main prediction endpoint. Accepts an audio file, performs preprocessing and inference,
# and returns the predicted class and probabilities.
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            probs, pred_idx = predict_file(filepath)
            os.remove(filepath)  # Clean up after prediction
            return jsonify({
                'prediction': CLASS_MAP.get(pred_idx, f'Class {pred_idx}'),
                'probabilities': {
                    CLASS_MAP.get(i, f'Class {i}'): round(p, 4)
                    for i, p in enumerate(probs)
                }
            })
        except Exception as e:
            return jsonify({'error': f'Inference failed: {str(e)}'}), 500

    return jsonify({'error': 'Unsupported file extension'}), 400

# ----------------------------
# Run the Flask app locally.
# ----------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
