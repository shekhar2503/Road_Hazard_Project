# 🛣️ Road Hazard Intelligence System

A road hazard detection and reporting platform with image inference, GPS-based reports, hazard-aware route recommendations, and a municipal dashboard. Powered by **YOLO11s (trained at 800×800)**, **FastAPI**, **PostgreSQL + PostGIS**, and **OSRM**. The frontend uses HTML, CSS, vanilla JavaScript, and Leaflet; the repository also includes Python training, evaluation, and command-line prediction scripts.

---

## 📌 Project Overview

The model, dataset configuration, and API use four hazard categories. Route recommendations rank candidate routes by distance and nearby active hazards:

| Class ID | Hazard Category | Description | Source Dataset | Production Ontology |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Pothole** | Structural cavities, depressions, and surface ruptures | RDD2022 | `pothole` |
| **1** | **Road Crack** | Longitudinal, transverse, and alligator cracks | RDD2022 | `road_crack` |
| **2** | **Waterlogging** | Standing water pools, puddles, and roadway flooding | FloodDET | `waterlogging` |
| **3** | **Construction Barrier** | Barricades, panels, and traffic cones | ROADWork (Experiment 3); Roadwork Cones (earlier pipeline) | `construction_barrier` |

---

## 🏗️ End-to-End System Architecture

```text
       User Upload (Image) + Browser GPS Location
               ↓
    FastAPI Inference Service (Port 5000: POST /detect/image)
               ↓
    YOLO11s Experiment 3 Detector (800×800, CUDA or CPU)
               ↓
    Frontend saves each detection via POST /hazards
               ↓
  Rule-Based Severity Engine (Low / Medium / High)
               ↓
  Maintenance Priority Engine (0–100 Score, Low / Medium / High / Critical)
               ↓
  PostgreSQL + PostGIS Database (geometry(Point, 4326))
               ↓
  Hazard-Aware Routing Engine (OSRM + PostGIS ST_DWithin Proximity Decay)
               ↓
  Frontend Dashboard & Interactive Leaflet Map (Port 8000)
  ├── User Detection Portal (detect.html)
  ├── Spatial Map & Route Planning (map.html)
  └── Municipal Authority Portal (index.html, admin-dashboard.html)
```

The detection page requests the device's current GPS coordinates before sending the image. Denied, unavailable, or timed-out location access stops that workflow. All detections from the image are reported at the device's current location, rather than a location inferred from the image. The `/detect/image` API itself only returns detections and an annotated image; persistence is a separate `/hazards` request.

Severity and priority are rule-based engineering estimates, not learned severity labels. The map groups nearby hazard reports and displays OSRM route recommendations. The authority frontend uses session login to access dashboards and report status management.

---

## 📊 Authoritative Model: Experiment 3 (YOLO11s at 800×800)

The production checkpoint is fine-tuned on the merged and class-balanced dataset at **800×800 resolution**:

* **Authoritative Checkpoint:** `runs/detect/experiment3_yolo11s_800/weights/best.pt`
* **Architecture:** YOLO11s (~9.46M parameters, 21.7 GFLOPs)
* **Input Resolution:** 800 × 800
* **Test Evaluation Metrics:**
  * **mAP@50:** **73.14%**
  * **mAP@50-95:** **49.62%**
  * **Precision:** **78.73%**
  * **Recall:** **68.57%**

### Model Evolution History

* **Baseline (YOLO11n, 640×640):** mAP@50: 61.70%, mAP@50-95: 34.93%
* **Experiment 2 (YOLO11s, 800×800):** Initial scale-up validation
* **Experiment 3 (YOLO11s, 800×800 — Final):** Expanded dataset, test mAP@50: **73.14%**

The Experiment 3 values are recorded checkpoint evaluation results from [`reports/experiment3_raw_evaluation.json`](reports/experiment3_raw_evaluation.json). Baseline figures are historical final-epoch validation values; its weights and training log are not included in the current branch. Waterlogging and barrier test splits changed between Experiments 2 and 3, so their metrics are measured on different test sets; see [`reports/experiment3_final_audit.txt`](reports/experiment3_final_audit.txt).

---

## 📁 Repository Structure

```text
Road_Hazard_Project/
├── backend/
│   ├── main.py                     # FastAPI application & lifespan loader
│   ├── database.py                 # SQLAlchemy engine & PostGIS connection
│   ├── init_db.py                  # Database & PostGIS initialization script
│   ├── migrate_db.py               # Existing database schema updates
│   ├── create_admin.py             # Administrator account creation script
│   ├── requirements.txt            # API, database, inference & test dependencies
│   ├── .env.example                # Database, model, routing & session settings
│   ├── models/
│   │   ├── hazard.py               # PostGIS Point hazard model
│   │   └── admin.py                # Admin user authentication model
│   ├── routes/
│   │   ├── hazards.py              # CRUD & spatial hazard endpoints
│   │   ├── routing.py              # Hazard-aware route recommendation
│   │   └── auth.py                 # Session authentication routes
│   ├── schemas/                    # Pydantic request/response schemas
│   ├── services/
│   │   ├── detector.py             # RoadHazardDetector (YOLO11s wrapper)
│   │   ├── severity.py             # Rule-based Severity Engine
│   │   ├── priority.py             # Maintenance Priority Scoring Engine
│   │   ├── routing.py              # OSRM driving service client
│   │   └── hazard_analyzer.py      # Spatial buffer decay & route penalty
│   └── tests/                      # Unit, integration & live end-to-end checks
├── frontend/
│   ├── index.html                  # Access Gateway (Public & Authority portal)
│   ├── detect.html                 # Public detection & image analysis interface
│   ├── map.html                    # Leaflet map & hazard-aware routing interface
│   ├── about.html                  # Project overview page
│   ├── admin-dashboard.html        # Municipal Authority dashboard & summary
│   ├── admin-reports.html          # Paginated hazard report queue
│   ├── admin-report-details.html   # Report detail view & status management
│   ├── app.js                      # Detection logic & database integration
│   ├── map.js                      # Spatial rendering & route polyline visualizer
│   ├── auth.js                     # Session management & gateway logic
│   ├── admin.js                    # Authority workflows & PATCH status
│   ├── map.css                     # Map interface styles
│   └── styles.css                  # UI design system & responsive styling
├── runs/
│   └── detect/
│       └── experiment3_yolo11s_800/ # Production model weights & plots
│           └── weights/best.pt     # Authoritative trained model checkpoint
├── configs/
│   └── data.yaml                   # YOLO dataset split & class ontology config
├── src/
│   ├── train_experiment3.py         # Experiment 3 training & evaluation
│   ├── train.py                    # Configurable training CLI
│   ├── evaluate.py                 # Validation/test evaluation CLI
│   ├── predict.py                  # Images/videos/folders/webcam inference CLI
│   ├── process_datasets.py         # Original RDD2022 & Roadwork Cones preprocessing
│   ├── process_flooddet.py         # Original FloodDET COCO conversion
│   └── prepare_dataset.py          # Original balanced dataset assembly
├── docs/
│   └── BACKEND_IMPLEMENTATION_GUIDE.md
├── reports/                        # Comprehensive test & evaluation reports
│   ├── final_end_to_end_testing_report.txt
│   ├── final_pre_submission_audit_report.txt
│   └── experiment3_integration_testing_report.txt
├── requirements.txt                # Root Python dependencies
└── README.md                       # Master system documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- Python 3.10+ (Python 3.11 recommended in the backend documentation)
- PostgreSQL 14+ with PostGIS 3.0+ extension
- NVIDIA CUDA GPU (optional for inference; required by `train_experiment3.py`)
- Internet access for map tiles and the configured OSRM routing service

### 2. Clone and Install

Run commands from the repository root, using an activated virtual environment:

```bash
git clone --branch main https://github.com/shekhar2503/Road_Hazard_Project.git
cd Road_Hazard_Project
python -m venv venv
```

Activate with `.\venv\Scripts\Activate.ps1` in Windows PowerShell, `venv\Scripts\activate.bat` in Windows Command Prompt, or `source venv/bin/activate` on Linux/macOS. Then install both dependency sets:

```bash
pip install -r requirements.txt -r backend/requirements.txt
```

### 3. Environment Configuration

Create `backend/.env` based on `backend/.env.example`:

```dotenv
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/road_hazard_db
MODEL_PATH=runs/detect/experiment3_yolo11s_800/weights/best.pt
OSRM_BASE_URL=https://router.project-osrm.org
SESSION_SECRET_KEY=REPLACE_WITH_A_GENERATED_SECRET_AT_LEAST_32_CHARACTERS_LONG
SESSION_COOKIE_SECURE=false
```

Set your actual database credentials and replace the session secret, for example with a value generated by:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

The backend refuses to start when `SESSION_SECRET_KEY` is missing or shorter than 32 characters. `SESSION_COOKIE_SECURE=false` supports local HTTP development; the example configuration specifies `true` for HTTPS production use. Relative `MODEL_PATH` values resolve from the working directory; the included Experiment 3 checkpoint is the default, with `backend/models/best.pt` as a fallback if the selected file is absent.

### 4. Database Initialization & Admin Setup

```bash
# Create the database if permitted, enable PostGIS, and create missing tables
python backend/init_db.py

# Create a municipal administrator account
python backend/create_admin.py --username admin
```

PostgreSQL must already be running with PostGIS installed. The configured account needs permission to create the database and enable the extension, or those steps must be completed beforehand. Admin creation prompts for a password of 12–256 characters and confirmation. For an existing schema that needs priority columns and recalculated priority scores, use `python backend/migrate_db.py`.

### 5. Running the Backend (Port 5000)

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 5000 --reload
```

- API Health Check: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)
- Swagger Documentation: [http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs)

### 6. Running the Frontend (Port 8000)

In a second terminal, from the repository root:

```bash
python -m http.server 8000 --directory frontend
```

- User Detection Portal: [http://127.0.0.1:8000/detect.html](http://127.0.0.1:8000/detect.html)
- Interactive Map & Routing: [http://127.0.0.1:8000/map.html](http://127.0.0.1:8000/map.html)
- Access Gateway / Authority: [http://127.0.0.1:8000/index.html](http://127.0.0.1:8000/index.html)
- About Page: [http://127.0.0.1:8000/about.html](http://127.0.0.1:8000/about.html)

Allow browser location access to use the detection/reporting flow. Frontend scripts target `localhost:5000` when opened on `localhost`, and `127.0.0.1:5000` otherwise. Use the local URLs above with matching hostnames for frontend and backend; deploying to another host requires adjusting those API URLs and the backend's allowed origins.

---

## 🗂️ Dataset Preparation and Model CLI

Datasets are excluded from Git. Inference uses the included checkpoint and your own input images; training and evaluation require separate datasets and annotations.

### Original Preprocessing and Balancing Pipeline

`process_datasets.py` expects YOLO-format RDD2022 splits under `Datasets/Potholes_cracks/RDD_SPLIT/{train,val,test}/{images,labels}` and Roadwork Cones images plus `annotations.parquet` under `Datasets/roadwork_cones_dataset/roadwork_cones_dataset`. It maps RDD source class `4` to pothole, classes `0`–`2` to road crack, and excludes repair class `3`. For Roadwork Cones, it maps only `cone` and `vertical_pannel` to construction barrier and excludes `roadworks` signs.

`process_flooddet.py` expects FloodDET images in `Datasets/FloodDET/FloodDET/{train,val,test}` and corresponding `train.json`, `val.json`, and `test.json` files in `Datasets/FloodDET/annotations_source`. It maps COCO category `45` to waterlogging, clamps boxes, and removes six predefined duplicate test images. Its `SRC_IMG_BASE`, `SRC_ANNO_BASE`, and `OUT_BASE` constants hardcode `F:\Road_Hazards` paths; adjust them for your checkout so conversion writes to your `Datasets/processed/FloodDET` directory.

Roadwork preprocessing uses `pandas.read_parquet`; install a Parquet engine separately because the dependency files do not include one:

```bash
pip install pyarrow
python src/process_datasets.py
python src/process_flooddet.py
python src/prepare_dataset.py --processed_base Datasets/processed --out_base Datasets/balanced_dataset --seed 42
```

The assembly script selects 2,250 crack-only training images approximately proportionally by country, retains all pothole-positive images, targets 799 RDD background images, selects up to 200 Roadwork background images, and retains FloodDET backgrounds. Its `train.txt` repeats waterlogging-positive images seven times and pothole-only images once or twice (approximately 1.5×); mixed pothole/crack images receive one entry. Pothole repetition uses `hash(stem)`, so it can vary across processes unless `PYTHONHASHSEED` is fixed. Processed validation and test splits are copied without oversampling.

`configs/data.yaml` points to `Datasets/balanced_dataset`, with `train.txt`, `images/val`, and `images/test`. This is the original pipeline, separate from the expanded Experiment 3 dataset.

### Experiment 3 Training

`src/train_experiment3.py` expects the separately prepared merged dataset at `F:\Road_Hazards\Datasets\experiment3_dataset\merged_dataset\data.yaml`. It requires CUDA, trains `yolo11s.pt` at 800×800 for up to 100 epochs with patience 20, chooses batch size 16 or 8 based on free GPU memory, and evaluates the best checkpoint on validation and test splits. Its workspace path is hardcoded, and it has no CLI path overrides; adjust that path before using another checkout.

```bash
python src/train_experiment3.py
```

The repository does not include an assembly script for the expanded Experiment 3 dataset. `prepare_dataset.py` produces the earlier balanced dataset and does not recreate it.

### Prediction, Evaluation, and Configurable Training

Specify the current checkpoint and 800-pixel input size explicitly: `predict.py` and `evaluate.py` still default to the older baseline checkpoint paths and 640-pixel inputs.

```bash
# Use an image, video, image folder, or --source 0 for a webcam
python src/predict.py --weights runs/detect/experiment3_yolo11s_800/weights/best.pt --imgsz 800 --source path/to/image.jpg

# Evaluate using your locally prepared Experiment 3 dataset configuration
python src/evaluate.py --weights runs/detect/experiment3_yolo11s_800/weights/best.pt --imgsz 800 --data Datasets/experiment3_dataset/merged_dataset/data.yaml --split both

# Train on the original balanced dataset using the configurable CLI
python src/train.py --data configs/data.yaml --model yolo11n.pt --epochs 50 --batch 32 --imgsz 640 --project runs/detect --name custom_run
```

Prediction defaults to confidence `0.25` and IoU `0.45`, and saves annotated results under `runs/predict/exp`; use `--project` and `--name` to change that location. Evaluation accepts `--split val`, `test`, or `both`, and prints overall and per-class metrics. Configurable training accepts `--patience`, `--device`, and `--workers`. Training and evaluation select GPU `0` when available or CPU otherwise; use a distinct run name to keep new experiments separate.

---

## 🧪 Automated Test Suite

Run the full automated test suite:

```bash
pytest backend/tests/ -v
```

Integration tests require the configured PostgreSQL/PostGIS database, Experiment 3 checkpoint, and local dataset fixtures; routing checks may also contact OSRM. `backend/tests/conftest.py` adds the project root to the Python import path.

**Recorded Verification Results:**

- **Unit Tests:** 46 / 46 PASSED
- **Integration Tests:** 17 / 17 PASSED
- **Total:** **63 / 63 PASSED (100% Success Rate)**

These are historical results documented in [`reports/experiment3_integration_testing_report.txt`](reports/experiment3_integration_testing_report.txt), not a new test run or a count of every test now in `backend/tests`.

With the backend running on port 5000 and the referenced Experiment 3 test images available, execute the live end-to-end flow test:

```bash
python backend/tests/test_e2e_flow.py
```

---

## 📜 Key Backend API Contracts

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Model status, compute device, class list, and DB connection |
| `POST` | `/detect/image` | YOLO11s image inference at 800×800 with bounding boxes & confidence |
| `POST` | `/hazards` | Persist detected hazard with PostGIS geometry, severity, and priority |
| `GET` | `/hazards` | List hazards with category, severity, priority, status filters, sorting, and pagination |
| `GET` | `/hazards/{id}` | Retrieve a single hazard report |
| `GET` | `/hazards/priority` | Dedicated municipal repair queue sorted by priority score |
| `PATCH` | `/hazards/{id}/status` | Update workflow lifecycle status (`active` / `resolved`) |
| `DELETE` | `/hazards/{id}` | Remove hazard record |
| `POST` | `/route/recommend` | OSRM candidate routes evaluated against PostGIS hazard buffer |
| `POST` | `/api/auth/login` | Municipal authority session authentication |
| `GET` | `/api/auth/session` | Validate active authority session |
| `POST` | `/api/auth/logout` | Terminate session and clear cookie |

The authority pages check the login session in the frontend. The current hazard CRUD routes depend on a database session and do not enforce administrator authentication. Spatial proximity queries are performed by the route recommendation service, rather than a bounding-box filter on `GET /hazards`.

---

## 🛠️ Technology Stack

* **AI & Computer Vision:** Ultralytics YOLO11s, PyTorch, OpenCV, CUDA
* **API Backend:** FastAPI, Starlette Session Middleware, Uvicorn, Pydantic v2
* **Geospatial Database:** PostgreSQL, PostGIS, SQLAlchemy 2.0, GeoAlchemy2, psycopg v3
* **Routing Engine:** Open Source Routing Machine (OSRM) driving API
* **Security & Auth:** Argon2 password hashing (pwdlib), HTTP-only cookies
* **Frontend Web App:** Semantic HTML5, Vanilla JavaScript, Leaflet.js, CSS Design Tokens
* **Testing:** Pytest, HTTPX, FastAPI TestClient

---

## 📜 License
The project declares the MIT License for educational and research use. This branch does not currently include a standalone `LICENSE` file with the full license text.
