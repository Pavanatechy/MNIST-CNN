# MNIST AI - Handwritten Digit Recognition System

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.14.1-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.13.4-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-98.97%25-10B981)](#cnn-architecture--performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An interactive, production-ready Deep Learning web platform that classifies handwritten digits (0–9) in real time using a Convolutional Neural Network (CNN) built with **TensorFlow / Keras 3**. Features a modern **HTML5 / CSS3 / JavaScript** AI dashboard with an interactive touch/mouse canvas, smart MNIST preprocessing, live confidence and 10-class probability visualization, and a RESTful **Flask** backend deployed with **Gunicorn** on **Render**.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Technology Stack](#technology-stack)
3. [Interactive Frontend Highlights](#interactive-frontend-highlights)
4. [Smart Image Preprocessing Pipeline](#smart-image-preprocessing-pipeline)
5. [CNN Architecture & Specifications](#cnn-architecture--specifications)
6. [Project Structure](#project-structure)
7. [Local Installation & Setup](#local-installation--setup)
8. [Running Locally](#running-locally)
9. [Automated Testing Suite](#automated-testing-suite)
10. [REST API Documentation](#rest-api-documentation)
11. [Render Cloud Deployment Guide](#render-cloud-deployment-guide)
12. [Git Version Control & Repository](#git-version-control--repository)

---

## Key Features

- **Trained CNN Deep Learning Model**: High-performing CNN reaching **98.97% test accuracy** on the canonical MNIST dataset (70,000 images).
- **Interactive Drawing Studio**: 280×280 pixel canvas with full mouse and touchscreen support, smooth anti-aliased strokes, and one-click clear.
- **Smart Preprocessing Pipeline**: Automatically detects stroke bounding boxes, scales preserving aspect ratio into a 20×20 box, centers within a 28×28 frame, inverts background to black, and normalizes pixels to `[0.0, 1.0]`.
- **Live "AI Vision" Preview**: Displays the exact 28×28 preprocessed image seen by the neural network in real time.
- **Instant Confidence & Prediction**: Bold typography displaying the predicted digit, confidence percentage, classification quality badge, and CPU inference latency (<15 ms).
- **10-Class Probability Visualization**: Animated Chart.js bar chart for all ten digits (0–9) highlighting the top prediction in electric violet/purple, with an accessible CSS fallback.
- **Test Sample Gallery**: Pre-loaded authentic MNIST test samples for digits 0 through 9, allowing one-click verification without drawing.
- **Live Health & Status Polling**: Asynchronously monitors the `/health` endpoint to display online status and network latency.
- **Cloud-Ready Deployment**: Configured with `render.yaml`, `Procfile`, Gunicorn WSGI server, and environment variables for seamless deployment on Render.

---

## Technology Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | HTML5, Modern CSS3 | Semantic / Responsive | Responsive layout, light theme with electric blue & purple accents |
| **Frontend Scripting** | Vanilla JavaScript (ES6+) | Modern Web APIs | Canvas drawing engine, coordinate mapping, touch events, API fetch |
| **Data Visualization** | Chart.js | 4.x (CDN) | Animated 10-class Softmax probability distribution bar chart |
| **Backend Framework** | Flask | 3.1.2 | Routing, template rendering, and REST API controllers |
| **CORS Middleware** | Flask-CORS | 6.0.2 | Cross-Origin Resource Sharing |
| **Deep Learning** | TensorFlow / Keras 3 | 2.21.0 / 3.14.1 | CNN model architecture, weights loading, execution |
| **Image Processing** | Pillow (PIL) | 12.2.0 | Base64 decoding, format conversions, image resizing |
| **Numerical Computing** | NumPy | 2.5.1 | High-performance array reshaping, validation, and math |
| **WSGI Server** | Gunicorn | 26.0.0 | Production-grade multi-threaded WSGI HTTP server |
| **Language Runtime** | Python | 3.13.4 | Core programming language |
| **Hosting Platform** | Render | PaaS | Automated Git-driven deployment and health monitoring |

---

## Interactive Frontend Highlights

### 1. Navigation Bar
- **Brand Identity**: `MNIST AI` logo with a gradient neural net icon and deep learning badge.
- **Section Anchors**: Smooth-scrolling links to *Home*, *Drawing Studio*, *Sample Testing*, *How It Works*, and *About Model*.
- **Live API Status Badge**: Color-coded pill (`API Online` / `API Offline`) that polls `/health` with ping latency indicators.

### 2. Hero Section
- High-impact typography: *"Turn Your Handwriting Into AI Predictions"*.
- Subtitle and call-to-action buttons (*"Start Recognition"* & *"Test MNIST Samples"*).
- Key performance metrics grid: **98.97% Accuracy**, **< 15 ms Latency**, **515K Parameters**, **10 Output Classes**.

### 3. Digit Drawing Studio
- **280×280 Canvas**: Clean white background with smooth black ink strokes.
- **Cross-Platform Input**: Full mouse event handling (`mousedown`, `mousemove`, `mouseup`) and mobile/tablet touch handling (`touchstart`, `touchmove`, `touchend`) with scroll-locking on canvas.
- **Controls**: *Predict Digit* (primary gradient button with loading state) and *Clear Canvas*.
- **28×28 AI Vision Preview**: Small real-time preview showing how the canvas image was centered and inverted for the neural network.

### 4. Prediction Results & Probabilities
- **Large Predicted Digit Display**: Eye-catching typography with pop-in animations.
- **Metrics Badges**: Confidence score (e.g., `99.98%`), classification status (*High Confidence Match*), and inference time.
- **Softmax Bar Chart**: Dynamic bar chart showcasing probability percentages across classes 0–9.

### 5. Sample Testing Gallery
- 10 interactive cards showcasing handwritten digits 0 through 9 from the official MNIST test set.
- Clicking any digit instantly loads it into the canvas, generates the preprocessed view, and triggers prediction.

### 6. Model Architecture & Specifications
- Comprehensive technical card detailing the optimizer (Adam), loss function, activations (ReLU & Softmax), and dataset split.
- Step-by-step visual diagram of the convolutional layers, pooling, flattening, and dense classifier.

---

## Smart Image Preprocessing Pipeline

Trained MNIST models require images with **black backgrounds (0.0)**, **white strokes (1.0)**, and digits **centered in a 20×20 box within a 28×28 matrix**.

The frontend and backend collaborate through a 4-step pipeline:

```
[User draws on 280×280 Canvas (White BG, Black Stroke)]
                      │
                      ▼
[Step 1: Bounding Box Detection]
  Scans pixel intensity (255 - grayscale) to find min/max X and Y coordinates.
                      │
                      ▼
[Step 2: Aspect-Ratio Preserving Scaling]
  Fits the extracted stroke bounding box into a 20×20 pixel region.
                      │
                      ▼
[Step 3: Centering & Inversion in 28×28 Canvas]
  Places the 20×20 digit at the geometric center of a 28×28 black canvas.
  Inverts pixel intensity: Canvas White (255) -> 0.0, Black Stroke (0) -> 1.0.
                      │
                      ▼
[Step 4: Inference Execution]
  Tensor shaped to (1, 28, 28, 1), passed to CNN model(img, training=False).
```

---

## CNN Architecture & Specifications

The CNN was trained on 60,000 MNIST training samples with Adam optimizer for 5 epochs:

```
Input Layer: (None, 28, 28, 1)
  │
  ├── Conv2D(32 filters, 3×3 kernel, ReLU)       ──> (None, 26, 26, 32)   [320 params]
  ├── MaxPooling2D(pool_size 2×2)                 ──> (None, 13, 13, 32)   [0 params]
  ├── Conv2D(64 filters, 3×3 kernel, ReLU)       ──> (None, 11, 11, 64)   [18,496 params]
  ├── Flatten()                                  ──> (None, 7744)         [0 params]
  ├── Dense(64 units, ReLU)                      ──> (None, 64)           [495,680 params]
  └── Dense(10 units, Softmax)                   ──> (None, 10)           [650 params]
```

### Performance Metrics
- **Trainable Parameters**: 515,146 (1.97 MB)
- **Keras 3 Model Artifact**: `saved_model/mnist_cnn.keras` (~6.2 MB)
- **TensorFlow Lite Artifact**: `saved_model/mnist_cnn.tflite` (~2.06 MB)
- **Test Set Loss**: 0.0348
- **Test Set Accuracy**: 98.97%
- **Inference Latency**: 8–15 ms on CPU

---

## Project Structure

```
DL Lab/
├── saved_model/
│   ├── mnist_cnn.keras        # Trained Keras 3 CNN model (primary)
│   └── mnist_cnn.tflite       # Optimized TensorFlow Lite model
│
├── templates/
│   └── index.html             # Modern HTML5 AI Dashboard
│
├── static/
│   ├── css/
│   │   └── style.css          # Responsive design, light theme, CSS variables
│   └── js/
│       ├── app.js             # Canvas engine, preprocessing, API client
│       └── samples.js         # Verified MNIST test samples (digits 0–9)
│
├── app.py                     # Flask REST API + Frontend Server
├── test_api.py                # Automated end-to-end test suite
├── mnist_samples.json         # Raw extracted sample matrices
├── mnist_training.ipynb       # Jupyter notebook with training pipeline
│
├── requirements.txt           # Pinned production dependencies
├── .python-version            # Python 3.13.4 specification for Render
├── Procfile                   # Process command for Gunicorn
├── render.yaml                # Infrastructure configuration for Render
├── .gitignore                 # Version control exclusions
└── README.md                  # Project documentation
```

---

## Local Installation & Setup

### 1. Prerequisites
- **Python 3.11, 3.12, or 3.13** installed
- **Git** installed

### 2. Clone the Repository
```bash
git clone https://github.com/Pavanatechy/MNIST-CNN.git
cd MNIST-CNN
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running Locally

Start the local Flask development server:

```bash
python app.py
```

Output:
```
Loading MNIST CNN model from: D:\DL Lab\saved_model\mnist_cnn.keras
Model loaded successfully into memory.
Model warmup complete. API is ready to serve requests.
Starting Flask server on http://0.0.0.0:5000 (debug=False)
 * Running on http://127.0.0.1:5000
```

Open **`http://127.0.0.1:5000`** in any web browser to interact with the dashboard.

---

## Automated Testing Suite

The repository includes a comprehensive automated test suite verifying all endpoints, frontend rendering, model predictions, and error handling:

```bash
python test_api.py
```

### Test Suite Output
```
=================================================================
 MNIST AI - Service Test Suite (Target: http://127.0.0.1:5000)
=================================================================

[1/5] Testing GET / (Frontend Dashboard) ...
   Status Code: 200
✅ GET / (Frontend) test passed successfully!

[2/5] Testing GET /?format=json & GET /api (API Metadata) ...
   Status Code: 200
✅ GET /api metadata test passed successfully!

[3/5] Testing GET /health ...
   Status Code: 200
   Response Body:
{
  "framework": "TensorFlow / Keras 3",
  "model": "mnist_cnn",
  "model_loaded": true,
  "status": "ok"
}
✅ GET /health test passed successfully!

[4/5] Testing POST /predict with 28x28 matrix sample ...
   Loaded sample #0 from MNIST test set (Ground Truth Label: 7)
   Status Code: 200
   Ground Truth Label:  7
   Predicted Digit:     7
   Confidence Score:    100.00%
   Server Inference:    21.78 ms
   🎯 Prediction accurately matches ground truth digit 7!
✅ POST /predict (matrix) test passed successfully!

[5/5] Testing error handling on invalid payloads ...
   Invalid dimensions status code: 400
✅ Error handling tests passed successfully!

=================================================================
 🎉 ALL API & FRONTEND TESTS PASSED SUCCESSFULLY!
=================================================================
```

---

## REST API Documentation

### 1. `GET /`
- **Default**: Serves the modern HTML5 frontend dashboard.
- **With `Accept: application/json` or `?format=json`**: Returns API specification and route index.

### 2. `GET /health`
Returns service and model availability status:
```json
{
  "framework": "TensorFlow / Keras 3",
  "model": "mnist_cnn",
  "model_loaded": true,
  "status": "ok"
}
```

### 3. `POST /predict`
Performs digit inference on an input image.

**Request Formats Supported:**
- **2D Numeric Matrix**: 28×28 list of floats `[0.0, 1.0]` or ints `[0, 255]`.
- **1D Numeric Array**: 784-element list of pixel values.
- **Base64 Encoded Image**: `data:image/png;base64,...` or raw base64 string.

**Example Request:**
```json
{
  "image": [[0.0, 0.0, ...], ...]
}
```

**Example Response:**
```json
{
  "prediction": 7,
  "confidence": 0.9998,
  "all_probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9998, 0.0, 0.0002],
  "inference_ms": 9.42
}
```

### 4. `GET /api/samples`
Returns 10 authentic handwritten sample digits (0–9) from the MNIST test set.

---

## Render Cloud Deployment Guide

This project is pre-configured for automated zero-downtime deployment on Render:

1. **Push to GitHub**:
   Ensure your latest code is pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Deploy modern interactive frontend and backend"
   git push origin main
   ```

2. **Create New Web Service on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com/).
   - Click **New +** -> **Web Service**.
   - Connect your repository (`https://github.com/Pavanatechy/MNIST-CNN.git`).

3. **Configure Settings**:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Environment Variables**:
     - `PYTHON_VERSION`: `3.13.4`
     - `TF_ENABLE_ONEDNN_OPTS`: `0`

4. **Deploy**:
   Click **Create Web Service**. Render will build the environment, load the CNN model, and serve both the web dashboard and REST API live!

---

## Git Version Control & Repository

Repository: [https://github.com/Pavanatechy/MNIST-CNN.git](https://github.com/Pavanatechy/MNIST-CNN.git)

To push the latest updates:
```bash
git add .
git commit -m "Add modern interactive AI frontend, canvas preprocessing, probability chart, and sample gallery"
git push origin main
```

---

## License

This project is licensed under the MIT License. Built with TensorFlow, Keras 3, and Flask.
