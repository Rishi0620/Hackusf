from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import boto3

# --- Custom Loss for loading model ---
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.5, alpha=0.35, reduction='sum_over_batch_size', name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = self.alpha * y_true * tf.pow(1 - y_pred, self.gamma) + \
                 (1 - self.alpha) * (1 - y_true) * tf.pow(y_pred, self.gamma)
        return tf.reduce_mean(weight * ce)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config

# --- Flask Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- AWS S3 Setup ---
S3_BUCKET = 'hackusf'
S3_KEY = 'final_balanced_model.keras'
LOCAL_MODEL_PATH = 'downloaded_model.keras'

# Download model from S3 (once)
def download_model_from_s3():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Downloading model from S3...")
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)
        print("Download complete.")

# Call this once at startup
download_model_from_s3()

# Load model
model = tf.keras.models.load_model(LOCAL_MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})

# --- Helpers ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        os.remove(filepath)

        label = "malignant" if prediction >= 0.5 else "benign"
        confidence = float(prediction) if label == "malignant" else 1 - float(prediction)

        return jsonify({
            'status': 'success',
            'prediction': {
                'class': label,
                'confidence': round(confidence * 100, 2),
                'probability': float(prediction)
            },
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)