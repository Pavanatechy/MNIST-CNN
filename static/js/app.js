/**
 * MNIST AI - Handwritten Digit Recognition System
 * Interactive Frontend Controller
 * Connects Canvas Drawing with Flask / TensorFlow Backend
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // 1. DOM Element References
    // -------------------------------------------------------------------------
    const drawingCanvas = document.getElementById('drawingCanvas');
    const previewCanvas = document.getElementById('previewCanvas');
    const canvasOverlay = document.getElementById('canvasOverlay');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const predictBtnText = document.getElementById('predictBtnText');

    const predictedDigitEl = document.getElementById('predictedDigit');
    const confidenceValueEl = document.getElementById('confidenceValue');
    const inferenceValueEl = document.getElementById('inferenceValue');
    const classificationStatusEl = document.getElementById('classificationStatus');
    const predictionBadgeEl = document.getElementById('predictionBadge');

    const apiStatusBadge = document.getElementById('apiStatusBadge');
    const fullApiStatus = document.getElementById('fullApiStatus');
    const fullApiStatusText = document.getElementById('fullApiStatusText');
    const healthPingEl = document.getElementById('healthPing');
    const refreshHealthBtn = document.getElementById('refreshHealthBtn');
    const toastEl = document.getElementById('toastNotification');

    // Canvas Contexts
    const ctx = drawingCanvas.getContext('2d');
    const prevCtx = previewCanvas.getContext('2d');

    // State Variables
    let isDrawing = false;
    let hasDrawn = false;
    let chartInstance = null;
    let lastX = 0;
    let lastY = 0;

    // -------------------------------------------------------------------------
    // 2. Canvas Initialization & High-DPI Setup
    // -------------------------------------------------------------------------
    function initCanvas() {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#000000';

        // Clear preview canvas
        prevCtx.fillStyle = '#000000';
        prevCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);

        hasDrawn = false;
        if (canvasOverlay) canvasOverlay.style.opacity = '0.45';
    }

    initCanvas();

    // -------------------------------------------------------------------------
    // 3. Coordinate Helper (Mouse & Touch)
    // -------------------------------------------------------------------------
    function getCanvasCoordinates(e) {
        const rect = drawingCanvas.getBoundingClientRect();
        const scaleX = drawingCanvas.width / rect.width;
        const scaleY = drawingCanvas.height / rect.height;

        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    // -------------------------------------------------------------------------
    // 4. Drawing Event Listeners (Mouse & Touch)
    // -------------------------------------------------------------------------
    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        hasDrawn = true;
        if (canvasOverlay) canvasOverlay.style.opacity = '0';

        const coords = getCanvasCoordinates(e);
        lastX = coords.x;
        lastY = coords.y;

        ctx.beginPath();
        ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#000000';
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const coords = getCanvasCoordinates(e);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();

        lastX = coords.x;
        lastY = coords.y;
    }

    function stopDrawing() {
        if (isDrawing) {
            isDrawing = false;
            ctx.closePath();
            updateLivePreview();
        }
    }

    // Mouse Events
    drawingCanvas.addEventListener('mousedown', startDrawing);
    drawingCanvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);
    drawingCanvas.addEventListener('mouseleave', stopDrawing);

    // Touch Events (Mobile/Tablet Support)
    drawingCanvas.addEventListener('touchstart', startDrawing, { passive: false });
    drawingCanvas.addEventListener('touchmove', draw, { passive: false });
    drawingCanvas.addEventListener('touchend', stopDrawing);
    drawingCanvas.addEventListener('touchcancel', stopDrawing);

    // -------------------------------------------------------------------------
    // 5. Canvas Controls (Clear & Reset)
    // -------------------------------------------------------------------------
    clearBtn.addEventListener('click', () => {
        initCanvas();
        resetPredictionUI();
        showToast('Canvas cleared', 'info');
    });

    function resetPredictionUI() {
        predictedDigitEl.textContent = '-';
        predictedDigitEl.classList.remove('highlight-anim');
        confidenceValueEl.textContent = '-';
        inferenceValueEl.textContent = '-';
        classificationStatusEl.textContent = 'Awaiting drawing';
        classificationStatusEl.className = 'meta-value text-muted';
        predictionBadgeEl.textContent = 'Ready';
        predictionBadgeEl.className = 'status-pill status-ready';

        if (chartInstance) {
            chartInstance.data.datasets[0].data = Array(10).fill(0);
            chartInstance.data.datasets[0].backgroundColor = Array(10).fill('rgba(99, 102, 241, 0.25)');
            chartInstance.update();
        }
        updateFallbackBars(Array(10).fill(0), -1);
    }

    // -------------------------------------------------------------------------
    // 6. Preprocessing Pipeline: Canvas (280x280) -> MNIST (28x28 Normalized)
    // -------------------------------------------------------------------------
    /**
     * Converts the 280x280 white canvas with black strokes into a 28x28
     * grayscale matrix adhering to the exact MNIST dataset standard:
     * - Isolated bounding box of the stroke
     * - Scaled to fit a 20x20 bounding box (preserving aspect ratio)
     * - Centered in a 28x28 frame
     * - Inverted to black background (0.0) with white stroke (1.0)
     */
    function preprocessCanvas() {
        const rawData = ctx.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height);
        const data = rawData.data;

        let minX = drawingCanvas.width, minY = drawingCanvas.height;
        let maxX = 0, maxY = 0;
        let nonWhitePixels = 0;

        // Step 1: Detect bounding box of non-white (drawn) strokes
        for (let y = 0; y < drawingCanvas.height; y++) {
            for (let x = 0; x < drawingCanvas.width; x++) {
                const idx = (y * drawingCanvas.width + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                // Invert intensity: 0 is white canvas, 255 is dark stroke
                const strokeIntensity = 255 - ((r + g + b) / 3);

                if (strokeIntensity > 30) {
                    nonWhitePixels++;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        // Check if canvas is virtually blank
        if (nonWhitePixels < 40 || maxX < minX || maxY < minY) {
            return null;
        }

        const boxWidth = maxX - minX + 1;
        const boxHeight = maxY - minY + 1;

        // Step 2: Fit within a 20x20 box preserving aspect ratio (standard MNIST)
        let scaledW, scaledH;
        if (boxWidth > boxHeight) {
            scaledW = 20;
            scaledH = Math.max(1, Math.round((boxHeight / boxWidth) * 20));
        } else {
            scaledH = 20;
            scaledW = Math.max(1, Math.round((boxWidth / boxHeight) * 20));
        }

        // Crop the stroke to a temporary canvas
        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = boxWidth;
        cropCanvas.height = boxHeight;
        const cropCtx = cropCanvas.getContext('2d');
        cropCtx.drawImage(drawingCanvas, minX, minY, boxWidth, boxHeight, 0, 0, boxWidth, boxHeight);

        // Scaled helper canvas for anti-aliased downsampling
        const scaledCanvas = document.createElement('canvas');
        scaledCanvas.width = scaledW;
        scaledCanvas.height = scaledH;
        const scaledCtx = scaledCanvas.getContext('2d');
        scaledCtx.drawImage(cropCanvas, 0, 0, boxWidth, boxHeight, 0, 0, scaledW, scaledH);

        // Step 3: Draw centered into a 28x28 black canvas
        const mnistCanvas = document.createElement('canvas');
        mnistCanvas.width = 28;
        mnistCanvas.height = 28;
        const mCtx = mnistCanvas.getContext('2d');

        // Black background
        mCtx.fillStyle = '#000000';
        mCtx.fillRect(0, 0, 28, 28);

        // Center coordinates inside 28x28 frame
        const offsetX = Math.round((28 - scaledW) / 2);
        const offsetY = Math.round((28 - scaledH) / 2);

        // Read cropped pixels and invert them (white stroke on black background)
        const scaledData = scaledCtx.getImageData(0, 0, scaledW, scaledH);
        const mData = mCtx.createImageData(scaledW, scaledH);

        for (let i = 0; i < scaledData.data.length; i += 4) {
            const r = scaledData.data[i];
            const g = scaledData.data[i + 1];
            const b = scaledData.data[i + 2];
            // Stroke intensity: darker original pixel becomes brighter white
            const strokeVal = Math.min(255, Math.max(0, 255 - Math.round((r + g + b) / 3)));
            
            mData.data[i] = strokeVal;
            mData.data[i + 1] = strokeVal;
            mData.data[i + 2] = strokeVal;
            mData.data[i + 3] = 255;
        }

        // Put inverted stroke onto 28x28 canvas
        mCtx.putImageData(mData, offsetX, offsetY);

        // Update the visible 28x28 AI Vision Preview canvas
        prevCtx.drawImage(mnistCanvas, 0, 0);

        // Step 4: Extract 28x28 matrix normalized to [0.0, 1.0]
        const finalImgData = mCtx.getImageData(0, 0, 28, 28).data;
        const matrix = [];

        for (let y = 0; y < 28; y++) {
            const row = [];
            for (let x = 0; x < 28; x++) {
                const idx = (y * 28 + x) * 4;
                // Grayscale intensity normalized to 0.0 - 1.0
                const val = finalImgData[idx] / 255.0;
                row.push(Number(val.toFixed(4)));
            }
            matrix.push(row);
        }

        return matrix;
    }

    function updateLivePreview() {
        preprocessCanvas();
    }

    // -------------------------------------------------------------------------
    // 7. Prediction Trigger & API Handler
    // -------------------------------------------------------------------------
    predictBtn.addEventListener('click', () => {
        executePrediction();
    });

    async function executePrediction() {
        const matrix = preprocessCanvas();

        if (!matrix) {
            showToast('Please draw a digit before predicting.', 'warning');
            return;
        }

        setPredictLoading(true);
        const startTime = performance.now();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ image: matrix })
            });

            const roundtripMs = Math.round(performance.now() - startTime);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.message || `Server returned HTTP ${response.status}`);
            }

            const data = await response.json();
            renderPredictionResults(data, roundtripMs);
            showToast(`Predicted digit ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`, 'success');

        } catch (error) {
            console.error('Prediction API Error:', error);
            showToast(`Prediction failed: ${error.message}`, 'error');
            predictionBadgeEl.textContent = 'Error';
            predictionBadgeEl.className = 'status-pill status-warning';
        } finally {
            setPredictLoading(false);
        }
    }

    function setPredictLoading(isLoading) {
        predictBtn.disabled = isLoading;
        if (isLoading) {
            predictBtnText.textContent = 'Analyzing...';
            predictionBadgeEl.textContent = 'Processing...';
            predictionBadgeEl.className = 'status-pill status-warning';
        } else {
            predictBtnText.textContent = 'Predict Digit';
        }
    }

    function renderPredictionResults(data, roundtripMs) {
        const digit = data.prediction;
        const confidence = data.confidence;
        const confPercent = (confidence * 100).toFixed(2);
        const inferenceMs = data.inference_ms;
        const allProbs = data.all_probs;

        // Display Predicted Digit with animation
        predictedDigitEl.textContent = digit;
        predictedDigitEl.classList.remove('highlight-anim');
        void predictedDigitEl.offsetWidth; // Trigger reflow
        predictedDigitEl.classList.add('highlight-anim');

        // Confidence
        confidenceValueEl.textContent = `${confPercent}%`;

        // Inference time
        inferenceValueEl.textContent = `${inferenceMs} ms (RTT: ${roundtripMs}ms)`;

        // Classification Status Badge
        if (confidence >= 0.85) {
            classificationStatusEl.textContent = 'High Confidence Match';
            classificationStatusEl.className = 'meta-value text-accent font-bold';
            predictionBadgeEl.textContent = 'High Confidence';
            predictionBadgeEl.className = 'status-pill status-success';
        } else if (confidence >= 0.50) {
            classificationStatusEl.textContent = 'Moderate Confidence';
            classificationStatusEl.className = 'meta-value font-bold';
            predictionBadgeEl.textContent = 'Moderate';
            predictionBadgeEl.className = 'status-pill status-warning';
        } else {
            classificationStatusEl.textContent = 'Ambiguous / Low Confidence';
            classificationStatusEl.className = 'meta-value text-muted';
            predictionBadgeEl.textContent = 'Uncertain';
            predictionBadgeEl.className = 'status-pill status-ready';
        }

        // Update Probabilities Chart
        updateProbabilityChart(allProbs, digit);
    }

    // -------------------------------------------------------------------------
    // 8. Probability Visualization (Chart.js + Accessible CSS Fallback)
    // -------------------------------------------------------------------------
    function initProbabilityChart() {
        const chartCanvas = document.getElementById('probabilityChart');
        if (!chartCanvas) return;

        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded. Using accessible CSS fallback bars.');
            document.getElementById('fallbackBars').style.display = 'flex';
            updateFallbackBars(Array(10).fill(0), -1);
            return;
        }

        const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        const initialData = Array(10).fill(0);

        const ctxChart = chartCanvas.getContext('2d');

        chartInstance = new Chart(ctxChart, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability',
                    data: initialData,
                    backgroundColor: Array(10).fill('rgba(99, 102, 241, 0.25)'),
                    borderColor: Array(10).fill('#6366f1'),
                    borderWidth: 1.5,
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 400,
                    easing: 'easeOutQuart'
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        titleFont: { family: 'Inter', size: 12, weight: '700' },
                        bodyFont: { family: 'JetBrains Mono', size: 12 },
                        padding: 10,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return `Probability: ${(context.raw * 100).toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            font: { family: 'JetBrains Mono', weight: '700', size: 12 },
                            color: '#475569'
                        }
                    },
                    y: {
                        min: 0,
                        max: 1.0,
                        grid: { color: '#e2e8f0', strokeDash: [3, 3] },
                        ticks: {
                            font: { family: 'JetBrains Mono', size: 10 },
                            color: '#94a3b8',
                            callback: function(value) {
                                return `${Math.round(value * 100)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    function updateProbabilityChart(probabilities, topDigit) {
        if (chartInstance) {
            const backgroundColors = probabilities.map((_, i) =>
                i === topDigit ? '#7c3aed' : 'rgba(99, 102, 241, 0.25)'
            );
            const borderColors = probabilities.map((_, i) =>
                i === topDigit ? '#6d28d9' : '#6366f1'
            );

            chartInstance.data.datasets[0].data = probabilities;
            chartInstance.data.datasets[0].backgroundColor = backgroundColors;
            chartInstance.data.datasets[0].borderColor = borderColors;
            chartInstance.update();
        }

        // Always update fallback bars for accessibility and resilience
        updateFallbackBars(probabilities, topDigit);
    }

    function updateFallbackBars(probabilities, topDigit) {
        const fallbackContainer = document.getElementById('fallbackBars');
        if (!fallbackContainer) return;

        let html = '';
        for (let i = 0; i < 10; i++) {
            const prob = probabilities[i] || 0;
            const percent = (prob * 100).toFixed(1);
            const isTop = (i === topDigit && prob > 0);
            html += `
                <div class="fb-row">
                    <span class="fb-label ${isTop ? 'text-accent' : ''}">${i}</span>
                    <div class="fb-track">
                        <div class="fb-fill ${isTop ? 'top-match' : ''}" style="width: ${percent}%;"></div>
                    </div>
                    <span class="fb-value ${isTop ? 'font-bold text-accent' : ''}">${percent}%</span>
                </div>
            `;
        }
        fallbackContainer.innerHTML = html;
    }

    initProbabilityChart();

    // -------------------------------------------------------------------------
    // 9. Sample Testing Gallery (MNIST Real Samples 0-9)
    // -------------------------------------------------------------------------
    function initSampleGallery() {
        if (!window.MNIST_SAMPLES) return;

        for (let digit = 0; digit < 10; digit++) {
            const canvasEl = document.getElementById(`sampleCanvas${digit}`);
            if (!canvasEl) continue;

            const sampleMatrix = window.MNIST_SAMPLES[digit];
            if (!sampleMatrix) continue;

            const sCtx = canvasEl.getContext('2d');
            const imgData = sCtx.createImageData(28, 28);

            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    const idx = (y * 28 + x) * 4;
                    const val = sampleMatrix[y][x]; // 0 to 255
                    imgData.data[idx] = val;
                    imgData.data[idx + 1] = val;
                    imgData.data[idx + 2] = val;
                    imgData.data[idx + 3] = 255;
                }
            }
            sCtx.putImageData(imgData, 0, 0);
        }
    }

    window.loadSampleDigit = function(digit) {
        if (!window.MNIST_SAMPLES || !window.MNIST_SAMPLES[digit]) {
            showToast(`Sample for digit ${digit} not available`, 'error');
            return;
        }

        const sample28 = window.MNIST_SAMPLES[digit];

        // 1. Render onto drawing canvas scaled up to 280x280 (as white background with black stroke)
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);

        // Draw the digit enlarged with anti-aliasing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tCtx = tempCanvas.getContext('2d');
        const tImgData = tCtx.createImageData(28, 28);

        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const idx = (y * 28 + x) * 4;
                const strokeIntensity = sample28[y][x]; // 0 (bg) to 255 (digit)
                // Invert for white drawing canvas: 255 - strokeIntensity
                const canvasPixel = 255 - strokeIntensity;
                tImgData.data[idx] = canvasPixel;
                tImgData.data[idx + 1] = canvasPixel;
                tImgData.data[idx + 2] = canvasPixel;
                tImgData.data[idx + 3] = 255;
            }
        }
        tCtx.putImageData(tImgData, 0, 0);

        // Scale up to 280x280 on main drawing canvas
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(tempCanvas, 0, 0, 280, 280);

        // 2. Render onto 28x28 preview canvas (white on black)
        prevCtx.fillStyle = '#000000';
        prevCtx.fillRect(0, 0, 28, 28);
        const pImgData = prevCtx.createImageData(28, 28);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const idx = (y * 28 + x) * 4;
                const val = sample28[y][x];
                pImgData.data[idx] = val;
                pImgData.data[idx + 1] = val;
                pImgData.data[idx + 2] = val;
                pImgData.data[idx + 3] = 255;
            }
        }
        prevCtx.putImageData(pImgData, 0, 0);

        // Hide canvas overlay
        if (canvasOverlay) canvasOverlay.style.opacity = '0';
        hasDrawn = true;

        // 3. Normalize to [0.0, 1.0] and send directly to /predict
        const normalizedMatrix = sample28.map(row => row.map(val => Number((val / 255.0).toFixed(4))));
        sendSamplePrediction(normalizedMatrix, digit);

        // Smooth scroll to studio to view results
        document.getElementById('studio').scrollIntoView({ behavior: 'smooth' });
    };

    async function sendSamplePrediction(matrix, expectedDigit) {
        setPredictLoading(true);
        const startTime = performance.now();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ image: matrix })
            });

            const roundtripMs = Math.round(performance.now() - startTime);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.message || `Server error: ${response.status}`);
            }

            const data = await response.json();
            renderPredictionResults(data, roundtripMs);
            showToast(`Loaded sample digit ${expectedDigit} • Predicted: ${data.prediction}`, 'success');

        } catch (error) {
            console.error('Sample Prediction Error:', error);
            showToast(`Prediction failed: ${error.message}`, 'error');
        } finally {
            setPredictLoading(false);
        }
    }

    initSampleGallery();

    // -------------------------------------------------------------------------
    // 10. API Health & Status Polling
    // -------------------------------------------------------------------------
    async function checkApiHealth() {
        const pingStart = performance.now();
        try {
            const response = await fetch('/health', {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });

            const latency = Math.round(performance.now() - pingStart);

            if (response.ok) {
                const data = await response.json();
                updateHealthStatusUI(true, data, latency);
            } else {
                updateHealthStatusUI(false, null, latency);
            }
        } catch (err) {
            updateHealthStatusUI(false, null, null);
        }
    }

    function updateHealthStatusUI(isOnline, data, latency) {
        if (isOnline) {
            apiStatusBadge.className = 'api-status-pill online';
            apiStatusBadge.querySelector('.status-label').textContent = 'API Online';

            if (fullApiStatus) {
                fullApiStatus.className = 'status-state online';
                fullApiStatusText.textContent = 'API Online & Ready';
            }
            if (healthPingEl && latency !== null) {
                healthPingEl.textContent = `${latency} ms`;
            }
        } else {
            apiStatusBadge.className = 'api-status-pill offline';
            apiStatusBadge.querySelector('.status-label').textContent = 'API Offline';

            if (fullApiStatus) {
                fullApiStatus.className = 'status-state offline';
                fullApiStatusText.textContent = 'API Offline';
            }
            if (healthPingEl) {
                healthPingEl.textContent = 'Unavailable';
            }
        }
    }

    if (refreshHealthBtn) {
        refreshHealthBtn.addEventListener('click', () => {
            showToast('Checking API health...', 'info');
            checkApiHealth();
        });
    }

    // Initial Health Check and periodic poll every 30 seconds
    checkApiHealth();
    setInterval(checkApiHealth, 30000);

    // -------------------------------------------------------------------------
    // 11. Toast Notifications Utility
    // -------------------------------------------------------------------------
    let toastTimeout = null;
    function showToast(message, type = 'info') {
        if (!toastEl) return;

        toastEl.textContent = message;
        toastEl.className = `toast show ${type}`;

        if (toastTimeout) clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            toastEl.className = 'toast';
        }, 3000);
    }
});
