# ClearScan Backend — Documentation

## System Overview (Plain Language)

The ClearScan backend is the brain of the entire system. When a radiographer taps **Assess** on their phone, here is exactly what happens:

1. The mobile app captures or selects a chest X-ray image.
2. The image is sent over the internet to this backend server.
3. The backend receives the image and feeds it to an AI quality classifier, which decides in under a second whether the X-ray is *Good*, *Fair*, or *Poor* quality.
4. If the X-ray is Fair or Poor, the backend automatically runs an image-enhancement algorithm to make it clearer.
5. The server sends back a quality score, a list of detected problems (blur, noise, low contrast), and — if enhanced — the improved image as well.
6. The mobile app displays all of this to the radiographer within 2–3 seconds of them tapping the button.

If the server cannot be reached (e.g. no internet at the clinic), the app automatically falls back to on-device analysis using the same algorithms running locally.

---

## System Overview (Technical)

**Framework:** FastAPI 0.110 with Uvicorn ASGI server.

**Four REST endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/health` | Liveness probe; reports model availability |
| `POST /api/v1/assess` | Quality classification (CNN or offline fallback) |
| `POST /api/v1/enhance/clahe` | Classical CLAHE+denoising+unsharp enhancement |
| `POST /api/v1/enhance/cnn` | U-Net autoencoder enhancement |
| `POST /api/v1/compare` | Runs both methods, returns winner by composite score |

**Model loading:** Both TensorFlow Keras models (`quality_classifier.keras`, `enhancement_autoencoder.keras`) are loaded at startup from `MODEL_DIR`. If either file is missing, the server starts in *offline mode* — the CLAHE pipeline and classical metric thresholds remain fully available.

**Online/offline fallback:** Every endpoint that uses the CNN classifier or enhancer catches `503 HTTPException` and gracefully falls back to classical methods, so the API never returns a failure due to missing model files.

**CLAHE pipeline:** `cv2.createCLAHE` → `cv2.fastNlMeansDenoising` → `skimage.filters.unsharp_mask`.

---

## Local Development

```bash
# 1. Clone the repo and enter the backend folder
git clone <repo-url>
cd clearscan-backend

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy .env.example to .env and edit as needed
cp .env.example .env

# 5. (Optional) copy model files
cp /path/to/quality_classifier.keras model_files/
cp /path/to/enhancement_autoencoder.keras model_files/

# 6. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API Reference

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/api/v1/health` | Liveness check | — | `HealthResponse` |
| POST | `/api/v1/assess` | Classify X-ray quality | `multipart/form-data` `file` | `QualityAssessmentResponse` |
| POST | `/api/v1/enhance/clahe` | CLAHE enhancement | `multipart/form-data` `file` | `EnhancementResponse` |
| POST | `/api/v1/enhance/cnn` | CNN enhancement | `multipart/form-data` `file` | `EnhancementResponse` |
| POST | `/api/v1/compare` | Compare both methods | `multipart/form-data` `file` | `ComparisonResponse` |

**Example curl commands:**

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Assess quality
curl -X POST http://localhost:8000/api/v1/assess \
  -F "file=@xray.png"

# Enhance with CLAHE
curl -X POST http://localhost:8000/api/v1/enhance/clahe \
  -F "file=@xray.png"

# Enhance with CNN
curl -X POST http://localhost:8000/api/v1/enhance/cnn \
  -F "file=@xray.png"

# Compare both methods
curl -X POST http://localhost:8000/api/v1/compare \
  -F "file=@xray.png"
```

---

## Model File Setup

The `.keras` model files are **not committed to git** (they are in `.gitignore`) because they can be hundreds of megabytes. You must place them manually:

```
model_files/
  quality_classifier.keras        # from clearscan-models pipeline, script 02
  enhancement_autoencoder.keras   # from clearscan-models pipeline, script 03
```

Copy from the models pipeline:
```bash
cp ../clearscan-models/models/saved/quality_classifier.keras model_files/
cp ../clearscan-models/models/saved/enhancement_autoencoder.keras model_files/
```

**On Render:** Use a Persistent Disk mounted at `/model_files`. Upload the files via the Render Shell or an scp command after first deploy.

**On Railway:** Use a Volume mounted at `/model_files`. Upload via the Railway CLI: `railway run scp quality_classifier.keras ...`.

---

## Deployment Guide (Render)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Web Service**.
3. Connect your GitHub repository.
4. Select **Docker** as the runtime (Render auto-detects the `Dockerfile`).
5. Set environment variables (or use the provided `render.yaml`):
   - `MODEL_DIR=/model_files`
   - `CORS_ORIGINS=*`
   - `LOG_LEVEL=INFO`
6. Add a **Persistent Disk**: mount path `/model_files`, size 1 GB.
7. Click **Create Web Service** and wait for the build (~5 minutes).
8. Once live, upload model files via the Render Shell tab.
9. Your deployment URL will be `https://clearscan-api.onrender.com`.

---

## Deployment Guide (Railway)

1. Push this repo to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub Repo**.
3. Railway detects the `Dockerfile` automatically.
4. In **Variables**, add: `PORT=8000`, `MODEL_DIR=/model_files`, `CORS_ORIGINS=*`.
5. Add a **Volume**: mount at `/model_files`.
6. Redeploy. Upload model files via `railway run` or the Railway shell.
7. Your deployment URL appears on the project dashboard.

---

## Offline Mode Architecture

The Flutter app uses a two-step connectivity check before deciding whether to call this API:

1. **Network check:** `connectivity_plus` verifies the device has a wifi or mobile data connection.
2. **Server ping:** The app calls `GET /api/v1/health` with a 5-second timeout. If this returns 200, the server is reachable.

Only if **both** checks pass does the app use the remote API. This is more reliable than checking general internet access because a device can be online but the server might be cold-starting or temporarily down. The health endpoint is lightweight (< 50ms) and does no model inference.

If the health check fails, the app falls back to on-device TFLite inference using the same model files packaged with the app bundle.

---

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_DIR` | string | `model_files` | Directory containing `.keras` model files |
| `MAX_IMAGE_SIZE_MB` | int | `10` | Maximum upload size in megabytes |
| `CORS_ORIGINS` | string | `*` | Comma-separated allowed origins for CORS |
| `LOG_LEVEL` | string | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Test Suite

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_health.py -v
```

| Test | What it validates |
|------|------------------|
| `test_health_returns_200` | Health endpoint returns 200 with `status` and `timestamp` keys |
| `test_assess_valid_image` | Valid PNG returns `quality_class` in `["Good","Fair","Poor"]` |
| `test_assess_invalid_extension` | `.exe` file returns 400 |
| `test_assess_oversized_file` | File > 10 MB returns 413 |
| `test_enhance_clahe_returns_base64` | CLAHE endpoint returns non-empty `enhanced_image_b64` |
| `test_compare_returns_winner` | Compare endpoint returns `winner` as `"clahe"` or `"cnn"` |

---

## Troubleshooting

### 1. Render cold start timeout (free tier)

Render's free tier hibernates after 15 minutes of inactivity. The first request after hibernation can take up to 50 seconds to respond while the container restarts. **Fix:** Configure the Flutter app to use a 60-second timeout on the health-check ping. Display a "Connecting..." spinner during this wait so the user knows the app is working.

### 2. TensorFlow OOM on free tier (512 MB RAM)

TensorFlow loads the full model graph into memory at startup. Running multiple workers doubles this. **Fix:** Ensure `--workers 1` in the Uvicorn command (already set in the `Dockerfile` CMD). Also confirm that no batch inference or warmup pass runs at startup — the models are lazy-loaded per request.

### 3. CORS errors in Flutter

If the Flutter app is running on a physical device with a different IP than `localhost`, CORS will block requests unless origins are explicitly allowed. **Fix:** Set `CORS_ORIGINS=*` in your `.env` file during development. In production, replace with your app's deployment domain.

### 4. `libGL` / `libEGL` error in Docker

`opencv-python-headless` still needs some GL libraries to import correctly in a slim container. **Fix:** The `Dockerfile` already installs `libgl1-mesa-glx libglib2.0-0 libsm6 libxext6`. If the error persists on a different base image, also add `libgl1` and `libgles2`.

### 5. `brisque` C extension build failure

`brisque` depends on `libsvm` which may fail to compile without system build tools. **Fix:** Either install system dependencies first (`apt-get install libsvm-dev`) or build without binary: `pip install brisque --no-binary brisque`. If BRISQUE continues to fail at runtime, the comparison module already catches all exceptions and returns `-1.0` (BRISQUE is excluded from the composite score in that case).
