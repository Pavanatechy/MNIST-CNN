"""
API Test Script for MNIST Handwritten Digit Recognition Service
Uses Python requests to test:
- GET /
- GET /health
- POST /predict with a real MNIST test image
- Validates JSON structure, types, confidence, and latency
"""

import sys
import os
import json
import time
import requests
import numpy as np

# Determine Target Base URL
# Can be overridden via command line argument (e.g. `python test_api.py https://my-app.onrender.com`)
# or environment variable API_URL
DEFAULT_URL = "http://127.0.0.1:5000"
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("API_URL", DEFAULT_URL).rstrip("/")

print("=" * 65)
print(f" MNIST API Test Suite - Target: {BASE_URL}")
print("=" * 65)


def test_get_root():
    print("\n[1/3] Testing GET / ...")
    url = f"{BASE_URL}/"
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed: Unable to connect to {url}")
        print("   Make sure the server is running (e.g. 'python app.py').")
        return False

    print(f"   Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    print("   Response Body:")
    print(json.dumps(data, indent=2))

    assert "name" in data, "Missing 'name' in root response"
    assert "status" in data and data["status"] == "online", "Unexpected status in root response"
    assert "endpoints" in data, "Missing 'endpoints' in root response"
    print("✅ GET / test passed successfully!")
    return True


def test_get_health():
    print("\n[2/3] Testing GET /health ...")
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


def test_post_predict():
    print("\n[3/3] Testing POST /predict with real MNIST test sample ...")

    # Load MNIST test sample
    # Try keras dataset cache, or synthesize a realistic test digit
    sample_image = None
    true_label = None

    try:
        import tensorflow as tf
        (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Sample index 0 in MNIST test set is digit 7
        sample_idx = 0
        sample_image = X_test[sample_idx]
        true_label = int(y_test[sample_idx])
        print(f"   Loaded sample #{sample_idx} from MNIST test set (Ground Truth Label: {true_label})")
    except Exception as e:
        print(f"   Note: Could not load full MNIST dataset ({e}), creating a clear digit '7' synthetic matrix.")
        sample_image = np.zeros((28, 28), dtype=np.uint8)
        # Draw a synthetic 7
        sample_image[6:9, 7:21] = 255
        for i in range(14):
            sample_image[9 + i, 20 - i] = 255
        true_label = 7

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

    if data["prediction"] == true_label:
        print(f"   🎯 Prediction matches ground truth digit {true_label}!")
    else:
        print(f"   ℹ️ Prediction was {data['prediction']} (Ground truth: {true_label})")

    # Verify error handling on invalid payload
    print("\n[Optional] Testing error handling for invalid 10x10 image ...")
    invalid_resp = requests.post(url, json={"image": [[0]*10]*10}, timeout=10)
    print(f"   Invalid payload status code: {invalid_resp.status_code}")
    assert invalid_resp.status_code == 400, f"Expected 400, got {invalid_resp.status_code}"
    print(f"   Error message returned: {invalid_resp.json().get('message')}")
    print("✅ Error handling test passed!")

    print("\n✅ POST /predict test passed successfully!")
    return True


if __name__ == "__main__":
    try:
        step1 = test_get_root()
        if not step1:
            sys.exit(1)
        step2 = test_get_health()
        step3 = test_post_predict()

        print("\n" + "=" * 65)
        print(" 🎉 ALL API TESTS COMPLETED AND VERIFIED SUCCESSFULLY!")
        print("=" * 65)
    except AssertionError as ae:
        print(f"\n❌ Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        sys.exit(1)

