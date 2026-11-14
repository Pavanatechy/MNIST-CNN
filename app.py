import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False

# Initialize Flask application
app = Flask(__name__)
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


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint: Provides API status, version, and documentation of available routes.
    """
    return jsonify({
        'name': 'MNIST Handwritten Digit Recognition API',
        'status': 'online',
        'version': '1.0.0',
        'framework': 'TensorFlow / Keras 3',
        'model': 'CNN (mnist_cnn.keras)',
        'endpoints': {
            'GET /': 'API root status and information',
            'GET /health': 'Health check and model status',
            'POST /predict': 'Predict handwritten digit from 28x28 image JSON'
        },
        'input_specification': {
            'endpoint': '/predict',
            'method': 'POST',
            'content_type': 'application/json',
            'body_schema': {
                'image': '28x28 2D list of pixel values (0-255 or 0.0-1.0) or 784-element 1D list'
            }
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


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint: Accepts a 28x28 image as JSON, runs CNN inference,
    and returns predicted digit, confidence, class probabilities, and latency.
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

    # 3. Validate and parse image data to numpy array
    try:
        img_array = np.array(raw_image, dtype=np.float32)
    except (ValueError, TypeError) as e:
        return jsonify({
            'error': 'Bad Request',
            'message': f"Invalid image data: could not parse numeric values. Details: {str(e)}"
        }), 400

    # 4. Check for NaN or Infinite values
    if np.isnan(img_array).any() or np.isinf(img_array).any():
        return jsonify({
            'error': 'Bad Request',
            'message': 'Image data contains NaN or Infinite values.'
        }), 400

    # 5. Validate image dimensions (expected 28x28 = 784 pixels)
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

    # 6. Normalize pixel values to [0.0, 1.0] if in [0, 255] range
    if np.max(img) > 1.0:
        img = img / 255.0

    # 7. Execute inference and benchmark latency
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
        'available_endpoints': ['GET /', 'GET /health', 'POST /predict']
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
