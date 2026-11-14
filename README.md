# MNIST Handwritten Digit Recognition API

A production-ready Deep Learning REST API that classifies handwritten digits (0–9) from 28×28 grayscale images using a Convolutional Neural Network (CNN) built with **TensorFlow / Keras 3** on **Python 3.13**, served via **Flask** and **Gunicorn**, and configured for zero-downtime deployment on **Render**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [CNN Architecture & Performance](#cnn-architecture--performance)
4. [Project Structure](#project-structure)
5. [Local Installation & Setup](#local-installation--setup)
6. [Running the Application Locally](#running-the-application-locally)
7. [API Endpoints & Specifications](#api-endpoints--specifications)
8. [Automated API Testing](#automated-api-testing)
9. [GitHub Configuration & Upload Instructions](#github-configuration--upload-instructions)
10. [Render Cloud Deployment Guide](#render-cloud-deployment-guide)

---

## Project Overview

This project provides an end-to-end Deep Learning classification service:
- **Model Training**: Trained on the canonical MNIST dataset (60,000 training images, 10,000 test images) achieving >99% test accuracy.
- **Model Formats**: Saved in modern native Keras 3 format (`.keras`) and converted to mobile-ready TensorFlow Lite (`.tflite`).
- **REST API**: Built with Flask, providing health monitoring, input validation (dimensions, types, NaN checks), auto-normalization, and sub-10ms CPU inference latency.
- **Cloud-Ready**: Configured with WSGI Gunicorn server, dynamic PORT binding (`0.0.0.0:$PORT`), `.python-version`, and Render Blueprint configuration (`render.yaml`).

---

## Technology Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.13.4 | Core runtime |
| **Deep Learning Framework** | TensorFlow & Keras 3 | 2.21.0 / 3.14.1 | CNN model architecture, training, and execution |
| **Model Formats** | Keras (`.keras`) & TFLite (`.tflite`) | Native Keras 3 / TFLite | Server and edge inference deployment |
| **API Framework** | Flask | 3.1.2 | RESTful HTTP API routing and request handling |
| **CORS Middleware** | Flask-CORS | 6.0.2 | Cross-Origin Resource Sharing for frontends |
| **WSGI Server** | Gunicorn | 26.0.0 | Production-grade HTTP application server |
| **Numerical Processing** | NumPy | 2.5.1 | Array reshaping, validation, and normalization |
| **HTTP Client Testing** | Requests | 2.34.2 | Automated integration and regression test suite |
| **Cloud Hosting** | Render | Cloud PaaS | Continuous build & deployment from GitHub |

---

## CNN Architecture & Performance

The neural network is a sequential Convolutional Neural Network tailored for 28×28 single-channel grayscale inputs:

```
Input: (28, 28, 1)
  │
  ├── Conv2D(32 filters, 3×3 kernel, ReLU)       ──> Output: (26, 26, 32)   [320 params]
  ├── MaxPooling2D(pool_size 2×2)                 ──> Output: (13, 13, 32)   [0 params]
  ├── Conv2D(64 filters, 3×3 kernel, ReLU)       ──> Output: (11, 11, 64)   [18,496 params]
  ├── Flatten()                                  ──> Output: (7744)         [0 params]
  ├── Dense(64 units, ReLU)                      ──> Output: (64)           [495,680 params]
  └── Dense(10 units, Softmax)                   ──> Output: (10)           [650 params]
```

### Model Metrics
- **Total Parameters**: 1,545,440 (5.90 MB)
- **Trainable Parameters**: 515,146 (1.97 MB)
- **Keras Model File Size (`mnist_cnn.keras`)**: ~6.2 MB
- **TFLite Model File Size (`mnist_cnn.tflite`)**: ~2.06 MB (65% compression)
- **Evaluation Accuracy**: ~99.1% on MNIST unseen test dataset
- **Inference Latency**: ~6–9 ms per digit on standard CPU

---

## Project Structure

```
DL Lab/
│
├── saved_model/
│   ├── mnist_cnn.keras        # Trained Keras 3 CNN model (loaded by Flask API)
│   └── mnist_cnn.tflite       # Optimized TensorFlow Lite export
│
├── app.py                     # Production Flask REST API backend
├── test_api.py                # Automated end-to-end API test suite (Python requests)
├── mnist_training.ipynb       # Jupyter notebook with training & TFLite conversion pipeline
│
├── requirements.txt           # Python dependencies pinned for cloud deployment
├── .python-version            # Specifies Python 3.13.4 runtime for Render
├── Procfile                   # Cloud process execution command for Gunicorn
├── render.yaml                # Render Blueprint infrastructure specification
├── .gitignore                 # Excludes cache/venv while preserving saved_model/
└── README.md                  # Complete documentation and deployment guide
```

---

## Local Installation & Setup

### 1. Prerequisites
- Python 3.13 (or 3.11 / 3.12)
- Git

### 2. Clone or Navigate to the Project
```bash
cd "C:\Users\ADMIN\Desktop\DL Lab"
```

### 3. Create and Activate Virtual Environment (Optional but Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application Locally

### Development Server (Flask)
```bash
python app.py
```
By default, the server will load `saved_model/mnist_cnn.keras`, perform a warmup pass, and listen at:
```
http://127.0.0.1:5000
```

### Production Server (Gunicorn — Linux / macOS / WSL)
```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 120
```

> **Note on Windows**: Gunicorn relies on POSIX system calls (`fcntl`/`fork`). On local Windows, use `python app.py`. In Render's Linux environment, Gunicorn runs natively as specified in `Procfile` and `render.yaml`.

---

## API Endpoints & Specifications

### 1. Root Endpoint: `GET /`
Provides service status, model metadata, and route documentation.

**Request:**
```bash
curl -X GET http://127.0.0.1:5000/
```

**Response (`200 OK`):**
```json
{
  "name": "MNIST Handwritten Digit Recognition API",
  "status": "online",
  "version": "1.0.0",
  "framework": "TensorFlow / Keras 3",
  "model": "CNN (mnist_cnn.keras)",
  "endpoints": {
    "GET /": "API root status and information",
    "GET /health": "Health check and model status",
    "POST /predict": "Predict handwritten digit from 28x28 image JSON"
  },
  "input_specification": {
    "endpoint": "/predict",
    "method": "POST",
    "content_type": "application/json",
    "body_schema": {
      "image": "28x28 2D list of pixel values (0-255 or 0.0-1.0) or 784-element 1D list"
    }
  }
}
```

---

### 2. Health Check: `GET /health`
Liveness and readiness probe for cloud monitoring.

**Request:**
```bash
curl -X GET http://127.0.0.1:5000/health
```

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "model": "mnist_cnn",
  "model_loaded": true,
  "framework": "TensorFlow / Keras 3"
}
```

---

### 3. Prediction Endpoint: `POST /predict`
Performs digit classification on an input image.

- **Content-Type**: `application/json`
- **Body Schema**: `{"image": [...]}` where `image` is a 28×28 2D array or 784-element 1D array.
- **Normalization**: Pixel values in the 0–255 range are automatically converted to 0.0–1.0.

**Sample Request (Python):**
```python
import requests

payload = {
    "image": [[0.0] * 28 for _ in range(28)]  # Replace with actual 28x28 pixel values
}
response = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(response.json())
```

**Sample Response (`200 OK`):**
```json
{
  "prediction": 7,
  "confidence": 1.0,
  "all_probs": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0
  ],
  "inference_ms": 7.42
}
```

### Error Responses
The API implements structured JSON error handling:
- **`400 Bad Request`**: Sent if `Content-Type` is not `application/json`, the `image` key is missing, dimensions differ from 784 pixels, or values contain NaN/Infinity.
  ```json
  {
    "error": "Bad Request",
    "message": "Invalid image dimensions. Expected 28x28 (784 pixels), but received array with size 100 and shape [10, 10]."
  }
  ```
- **`404 Not Found`**: Sent if an unmapped route is requested.
  ```json
  {
    "error": "Not Found",
    "message": "The requested URL was not found on this server.",
    "available_endpoints": ["GET /", "GET /health", "POST /predict"]
  }
  ```

---

## Automated API Testing

The project includes `test_api.py`, which validates the full request/response lifecycle using Python `requests` and a real test sample from MNIST:

```bash
# Make sure your app is running in one terminal (python app.py)
# Then run the test suite in another terminal:
python test_api.py
```

To test against a remote cloud URL (such as your deployed Render app):
```bash
python test_api.py https://your-app-name.onrender.com
```

### Test Suite Execution Output
```
=================================================================
 MNIST API Test Suite - Target: http://127.0.0.1:5000
=================================================================

[1/3] Testing GET / ...
   Status Code: 200
✅ GET / test passed successfully!

[2/3] Testing GET /health ...
   Status Code: 200
✅ GET /health test passed successfully!

[3/3] Testing POST /predict with real MNIST test sample ...
   Loaded sample #0 from MNIST test set (Ground Truth Label: 7)
   Status Code: 200
   Ground Truth Label:  7
   Predicted Digit:     7
   Confidence Score:    100.00%
   Server Inference:    9.04 ms
   Total Roundtrip:     22.82 ms
   🎯 Prediction matches ground truth digit 7!

[Optional] Testing error handling for invalid 10x10 image ...
   Invalid payload status code: 400
✅ Error handling test passed!

✅ POST /predict test passed successfully!
=================================================================
 🎉 ALL API TESTS COMPLETED AND VERIFIED SUCCESSFULLY!
=================================================================
```

---

## GitHub Configuration & Upload Instructions

Follow these exact steps to push this project to GitHub. **(Do not push without verifying your repository details)**:

### Step 1: Initialize Git Repository
In your project directory (`C:\Users\ADMIN\Desktop\DL Lab`):
```bash
git init
```

### Step 2: Stage Project Files
```bash
git add .
```
Verify the staged files:
```bash
git status
```
> Notice that `saved_model/mnist_cnn.keras` (~6.2 MB) and `saved_model/mnist_cnn.tflite` (~2.06 MB) are properly staged, while virtual environments and caches are excluded by `.gitignore`.

### Step 3: Create Initial Commit
```bash
git commit -m "feat: complete MNIST CNN Flask API ready for Render cloud deployment"
```

### Step 4: Create a New GitHub Repository
1. Go to [GitHub New Repository](https://github.com/new).
2. Set Repository name to `mnist-digit-recognition-api` (or your preferred name).
3. Set visibility to **Public** (recommended for free Render deployment).
4. Leave "Add a README", ".gitignore", and "License" **unchecked** (we already have them).
5. Click **Create repository**.

### Step 5: Link Remote and Push
```bash
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/mnist-digit-recognition-api.git
git push -u origin main
```

*(Alternatively, if you use the GitHub CLI `gh` already installed on your machine):*
```bash
gh repo create mnist-digit-recognition-api --public --source=. --remote=origin --push
```

---

## Render Cloud Deployment Guide

Render will build and serve your API directly from your GitHub repository.

### Step 1: Create a Render Account
1. Visit [https://render.com](https://render.com) and sign up (or sign in with your GitHub account).

### Step 2: Create a New Web Service
1. In the Render Dashboard, click **New +** in the top navigation bar.
2. Select **Web Service**.
3. Choose **Build and deploy from a Git repository**.
4. Connect your GitHub account and select your `mnist-digit-recognition-api` repository.

### Step 3: Configure Service Settings
Fill in the following fields in the Render dashboard:

| Setting | Value | Explanation |
| :--- | :--- | :--- |
| **Name** | `mnist-digit-recognition-api` | Your unique service name |
| **Region** | `Oregon (US West)` or nearest | Cloud region |
| **Branch** | `main` | Git deployment branch |
| **Root Directory** | *(leave blank)* | Root of the repo |
| **Runtime** | `Python 3` | Python runtime |
| **Build Command** | `pip install -r requirements.txt` | Installs dependencies |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120` | Production WSGI command |
| **Instance Type** | `Free` | Free tier (512 MB RAM) |

### Step 4: Configure Environment Variables (Optional but Recommended)
Under **Environment Variables**, add:
- `PYTHON_VERSION`: `3.13.4`
- `TF_ENABLE_ONEDNN_OPTS`: `0`

*(If you use Render Blueprints, Render will automatically detect and apply `render.yaml` with these exact settings).*

### Step 5: Deploy the Service
1. Click **Create Web Service**.
2. Render will trigger the initial build:
   - Clones your repository.
   - Installs Python 3.13.
   - Runs `pip install -r requirements.txt`.
   - Starts the application using Gunicorn.
3. Once the build completes, the logs will show:
   ```
   ==> Starting service with 'gunicorn app:app --bind 0.0.0.0:$PORT ...'
   Loading MNIST CNN model from: .../saved_model/mnist_cnn.keras
   Model loaded successfully into memory.
   Model warmup complete. API is ready to serve requests.
   ==> Your service is live 🎉
   ```

### Step 6: Test Your Public Render URL
Your public service URL will be displayed at the top of the Render dashboard:
```
https://mnist-digit-recognition-api-xxxx.onrender.com
```

Test it immediately using `test_api.py`:
```bash
python test_api.py https://mnist-digit-recognition-api-xxxx.onrender.com
```
Or with `curl`:
```bash
curl -X GET https://mnist-digit-recognition-api-xxxx.onrender.com/health
```

> **Note on Free Tier Inactivity**: Free Render instances automatically spin down into a sleep state after 15 minutes of inactivity. The first incoming request will take ~30–50 seconds to cold-start the container and load the model into memory. Subsequent requests will execute in milliseconds.

