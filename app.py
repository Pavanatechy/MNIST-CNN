import os
import time
import json
import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

try:
    from PIL import Image
    pillow_available = True
except ImportError:
    pillow_available = False

try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False

# Initialize Flask application
app = Flask(__name__, template_folder='templates', static_folder='static')
if cors_available:
    CORS(app)

# Load trained MNIST CNN model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model', 'mnist_cnn.keras')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model file not found at: {MODEL_PATH}. "
        "Please ensure 'mnist_cnn.keras' exists in the 'saved_model' directory."
    )

print(f"Loading MNIST CNN model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully into memory.")

# Warm-up model to compile execution graph
_warmup_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
_ = model(_warmup_input, training=False)
print("Model warmup complete. API is ready to serve requests.")

# Load pre-extracted MNIST samples for quick testing if available
SAMPLES_FILE = os.path.join(BASE_DIR, 'mnist_samples.json')
sample_digits_cache = {}
if os.path.exists(SAMPLES_FILE):
    try:
        with open(SAMPLES_FILE, 'r') as f:
            sample_digits_cache = json.load(f)
    except Exception as e:
        print(f"Note: Could not preload mnist_samples.json: {e}")


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint:
    - Serves modern interactive frontend (index.html) for web browsers and default requests.
    - If explicitly requested with Accept: application/json or ?format=json, returns JSON API spec.
    """
    if request.headers.get('Accept') == 'application/json' or request.args.get('format') == 'json':
        return jsonify({
            'name': 'MNIST AI - Handwritten Digit Recognition API',
            'status': 'online',
            'version': '2.0.0',
            'framework': 'TensorFlow / Keras 3',
            'model': 'CNN (mnist_cnn.keras)',
            'endpoints': {
                'GET /': 'Interactive Frontend (HTML) or API metadata (JSON)',
                'GET /health': 'Health check and model status',
                'POST /predict': 'Predict handwritten digit from 28x28 image array or base64',
                'GET /api/samples': 'Retrieve verified MNIST test sample images (digits 0-9)'
            },
            'input_specification': {
                'endpoint': '/predict',
                'method': 'POST',
                'content_type': 'application/json',
                'body_schema': {
                    'image': '28x28 2D list of pixel values (0-255 or 0.0-1.0), 784-element 1D list, or base64 data URI'
                }
            }
        }), 200

    return render_template('index.html')


@app.route('/api', methods=['GET'])
def api_info():
    """
    Explicit JSON documentation route for API clients.
    """
    return jsonify({
        'name': 'MNIST AI - Handwritten Digit Recognition API',
        'status': 'online',
        'version': '2.0.0',
        'framework': 'TensorFlow / Keras 3',
        'model': 'CNN (mnist_cnn.keras)',
        'endpoints': {
            'GET /': 'Interactive Frontend',
            'GET /api': 'API documentation (this route)',
            'GET /health': 'Health check and model status',
            'POST /predict': 'Predict handwritten digit',
            'GET /api/samples': 'MNIST test samples (digits 0-9)'
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint: Returns service health status and model availability.
    """
    return jsonify({
        'status': 'ok',
        'model': 'mnist_cnn',
        'model_loaded': model is not None,
        'framework': 'TensorFlow / Keras 3'
    }), 200


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """
    Returns verified MNIST test sample digits (0-9) as JSON.
    """
    return jsonify({
        'status': 'ok',
        'samples': sample_digits_cache
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint:
    Accepts:
      - 28x28 2D list of numbers
      - 784-element 1D list of numbers
      - Base64 image string (data:image/png;base64,... or raw base64)
    Runs CNN inference and returns predicted digit, confidence, all 10 class probabilities, and latency.
    """
    # 1. Validate Content-Type
    if not request.is_json:
        return jsonify({
            'error': 'Bad Request',
            'message': 'Content-Type must be application/json'
        }), 400

    # 2. Parse JSON payload
    data = request.get_json(silent=True)
    if data is None or 'image' not in data:
        return jsonify({
            'error': 'Bad Request',
            'message': "Missing 'image' field in JSON request body."
        }), 400

    raw_image = data['image']

    # 3. Handle base64 image strings if provided
    if isinstance(raw_image, str):
        if not pillow_available:
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'Pillow is required for server-side base64 image decoding.'
            }), 500

        try:
            base64_data = raw_image
            if ',' in base64_data:
                base64_data = base64_data.split(',', 1)[1]
            image_bytes = base64.b64decode(base64_data)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            if pil_img.size != (28, 28):
                pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(pil_img, dtype=np.float32)

            # Invert if the image has a predominantly white background (like canvas default)
            # MNIST expects black background (0) and white stroke (255)
            border_mean = np.mean([
                img_array[0, :], img_array[-1, :],
                img_array[:, 0], img_array[:, -1]
            ])
            if border_mean > 127.5:
                img_array = 255.0 - img_array

        except Exception as e:
            return jsonify({
                'error': 'Bad Request',
                'message': f"Failed to process base64 image string: {str(e)}"
            }), 400
    else:
        # 4. Validate and parse numeric array data
        try:
            img_array = np.array(raw_image, dtype=np.float32)
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'Bad Request',
                'message': f"Invalid image data: could not parse numeric values. Details: {str(e)}"
            }), 400

    # 5. Check for NaN or Infinite values
    if np.isnan(img_array).any() or np.isinf(img_array).any():
        return jsonify({
            'error': 'Bad Request',
            'message': 'Image data contains NaN or Infinite values.'
        }), 400

    # 6. Validate image dimensions (expected 28x28 = 784 pixels)
    if img_array.size != 784:
        return jsonify({
            'error': 'Bad Request',
            'message': (
                f"Invalid image dimensions. Expected 28x28 (784 pixels), "
                f"but received array with size {img_array.size} and shape {list(img_array.shape)}."
            )
        }), 400

    try:
        img = img_array.reshape(1, 28, 28, 1)
    except ValueError as e:
        return jsonify({
            'error': 'Bad Request',
            'message': f"Could not reshape input to (1, 28, 28, 1): {str(e)}"
        }), 400

    # 7. Normalize pixel values to [0.0, 1.0] if in [0, 255] range
    if np.max(img) > 1.0:
        img = img / 255.0

    # 8. Execute inference and benchmark latency
    start_time = time.perf_counter()
    probs_tensor = model(img, training=False)
    probs = probs_tensor.numpy()[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    prediction = int(np.argmax(probs))
    confidence = float(np.max(probs))
    all_probs = [round(float(p), 4) for p in probs]

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence, 4),
        'all_probs': all_probs,
        'inference_ms': round(elapsed_ms, 2)
    }), 200


@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({
        'error': 'Bad Request',
        'message': getattr(e, 'description', str(e))
    }), 400


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested URL was not found on this server.',
        'available_endpoints': ['GET /', 'GET /health', 'POST /predict', 'GET /api/samples']
    }), 404


@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({
        'error': 'Method Not Allowed',
        'message': f"The HTTP method {request.method} is not allowed for the requested URL."
    }), 405


@app.errorhandler(500)
def handle_internal_server_error(e):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred during request processing.'
    }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"Starting Flask server on http://0.0.0.0:{port} (debug={debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
