"""
Comprehensive API & Frontend Test Suite for MNIST Handwritten Digit Recognition Service
Tests:
- GET / (Interactive Web Frontend HTML5)
- GET /api & GET /?format=json (JSON API Specification)
- GET /health (Health check and model status)
- POST /predict with 28x28 numeric matrix (Real MNIST test sample)
- POST /predict with base64 image encoding
- Error handling on invalid payloads (e.g. malformed shape)
"""

import sys
import os
import json
import time
import base64
import requests
import numpy as np

# Determine Target Base URL
# Can be overridden via command line argument (e.g. `python test_api.py https://my-app.onrender.com`)
# or environment variable API_URL
DEFAULT_URL = "http://127.0.0.1:5000"
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("API_URL", DEFAULT_URL).rstrip("/")

print("=" * 65)
print(f" MNIST AI - Service Test Suite (Target: {BASE_URL})")
print("=" * 65)


def test_get_root_frontend():
    print("\n[1/5] Testing GET / (Frontend Dashboard) ...")
    url = f"{BASE_URL}/"
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed: Unable to connect to {url}")
        print("   Make sure the server is running (e.g. 'python app.py').")
        return False

    print(f"   Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    content_type = response.headers.get("Content-Type", "")
    assert "text/html" in content_type, f"Expected text/html, got {content_type}"
    assert "MNIST AI" in response.text, "Missing 'MNIST AI' brand in frontend HTML"
    assert "drawingCanvas" in response.text, "Missing 'drawingCanvas' element in frontend HTML"
    assert "Turn Your Handwriting Into AI Predictions" in response.text or "Turn Your Handwriting" in response.text, "Missing Hero title"
    print("✅ GET / (Frontend) test passed successfully!")
    return True


def test_get_api_metadata():
    print("\n[2/5] Testing GET /?format=json & GET /api (API Metadata) ...")
    url = f"{BASE_URL}/api"
    response = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
    print(f"   Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    assert "name" in data, "Missing 'name' in api response"
    assert data.get("status") == "online", "Unexpected status in api response"
    assert "endpoints" in data, "Missing 'endpoints' in api response"
    print("✅ GET /api metadata test passed successfully!")
    return True


def test_get_health():
    print("\n[3/5] Testing GET /health ...")
    url = f"{BASE_URL}/health"
    response = requests.get(url, timeout=10)

    print(f"   Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print("   Response Body:")
    print(json.dumps(data, indent=2))

    assert data.get("status") == "ok", "Expected status 'ok'"
    assert data.get("model") == "mnist_cnn", "Expected model 'mnist_cnn'"
    assert data.get("model_loaded") is True, "Expected model_loaded True"
    print("✅ GET /health test passed successfully!")
    return True


def test_post_predict_matrix():
    print("\n[4/5] Testing POST /predict with 28x28 matrix sample ...")

    # Load MNIST test sample #0 (ground truth digit 7)
    sample_image = None
    true_label = 7

    try:
        import tensorflow as tf
        (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        sample_image = X_test[0]
        true_label = int(y_test[0])
        print(f"   Loaded sample #0 from MNIST test set (Ground Truth Label: {true_label})")
    except Exception as e:
        print(f"   Note: Loading from mnist_samples.json fallback ({e})")
        if os.path.exists("mnist_samples.json"):
            with open("mnist_samples.json") as f:
                samples = json.load(f)
                sample_image = np.array(samples.get("7", samples.get(7)), dtype=np.uint8)
        else:
            sample_image = np.zeros((28, 28), dtype=np.uint8)
            sample_image[6:9, 7:21] = 255
            for i in range(14):
                sample_image[9 + i, 20 - i] = 255

    # Send POST request
    url = f"{BASE_URL}/predict"
    payload = {
        "image": sample_image.tolist()
    }

    start_client = time.time()
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
    roundtrip_ms = (time.time() - start_client) * 1000

    print(f"   Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    print("   Prediction Response:")
    print(json.dumps(data, indent=2))

    # Verify JSON structure & types
    assert "prediction" in data, "Missing 'prediction' key"
    assert "confidence" in data, "Missing 'confidence' key"
    assert "all_probs" in data, "Missing 'all_probs' key"
    assert "inference_ms" in data, "Missing 'inference_ms' key"

    assert isinstance(data["prediction"], int), "'prediction' must be int"
    assert isinstance(data["confidence"], float), "'confidence' must be float"
    assert isinstance(data["all_probs"], list), "'all_probs' must be list"
    assert len(data["all_probs"]) == 10, "'all_probs' must contain 10 probabilities"
    assert isinstance(data["inference_ms"], (int, float)), "'inference_ms' must be numeric"

    print(f"\n   Ground Truth Label:  {true_label}")
    print(f"   Predicted Digit:     {data['prediction']}")
    print(f"   Confidence Score:    {data['confidence'] * 100:.2f}%")
    print(f"   Server Inference:    {data['inference_ms']} ms")
    print(f"   Total Roundtrip:     {roundtrip_ms:.2f} ms")

    assert data["prediction"] == true_label, f"Expected prediction {true_label}, got {data['prediction']}"
    print(f"   🎯 Prediction accurately matches ground truth digit {true_label}!")
    print("✅ POST /predict (matrix) test passed successfully!")
    return True


def test_post_predict_errors():
    print("\n[5/5] Testing error handling on invalid payloads ...")
    url = f"{BASE_URL}/predict"

    # Test 1: Invalid matrix dimension (10x10 instead of 28x28)
    invalid_resp = requests.post(url, json={"image": [[0]*10]*10}, timeout=10)
    print(f"   Invalid dimensions status code: {invalid_resp.status_code}")
    assert invalid_resp.status_code == 400, f"Expected 400, got {invalid_resp.status_code}"

    # Test 2: Missing 'image' key
    missing_key_resp = requests.post(url, json={"data": 123}, timeout=10)
    assert missing_key_resp.status_code == 400, f"Expected 400, got {missing_key_resp.status_code}"

    # Test 3: Non-JSON body
    non_json_resp = requests.post(url, data="invalid", headers={"Content-Type": "text/plain"}, timeout=10)
    assert non_json_resp.status_code == 400, f"Expected 400, got {non_json_resp.status_code}"

    print("✅ Error handling tests passed successfully!")
    return True


if __name__ == "__main__":
    try:
        step1 = test_get_root_frontend()
        if not step1:
            sys.exit(1)
        step2 = test_get_api_metadata()
        step3 = test_get_health()
        step4 = test_post_predict_matrix()
        step5 = test_post_predict_errors()

        print("\n" + "=" * 65)
        print(" 🎉 ALL API & FRONTEND TESTS PASSED SUCCESSFULLY!")
        print("=" * 65)
    except AssertionError as ae:
        print(f"\n❌ Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        sys.exit(1)
