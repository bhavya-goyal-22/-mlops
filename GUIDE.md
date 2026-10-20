# Teaching guide

This file is the long explanation. `README.md` is the short command list. Read this when you want to know what each file does and how to start the whole pipeline yourself.

Run every command from the project folder:

```bash
cd /home/hitech/Downloads/assignments/mlops-platform
```

## What this project does

It predicts whether a patient has heart disease (yes or no) from the UCI Heart Disease dataset (id 45). The path from raw data to a monitored API is:

1. Download the CSV if it is not already on disk.
2. Drop rows that are missing `ca` or `thal`, and turn the target into 0 or 1.
3. Draw EDA charts (class balance, histograms, correlation).
4. Train two models (logistic regression and random forest), pick the one with the higher cross-validated ROC-AUC, and save it as ONNX.
5. Log every run in MLflow.
6. Serve that ONNX file with FastAPI.
7. Put the API in Docker, then in local Kubernetes (Minikube), with Prometheus and Grafana watching it.

`prediction` is `0` (no disease) or `1` (disease). `probability` is the model’s confidence that the class is 1.

## Start here (one-time setup)

You need Python 3.12, and later Docker plus Minikube and kubectl if you want the container and cluster steps.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`source venv/bin/activate` turns the virtual environment on. Your prompt usually shows `(venv)`. Do that in every new terminal before the Python commands below. `requirements.txt` is the only dependency file. Training, tests, CI, and Docker all install that same file.

`PYTHONPATH=src` tells Python to look for the `mlops` package inside `src/`. Without it, `import mlops` fails.

## Order to start things

Do them in this order the first time. Later you can skip steps you already finished.

| Step | What you start | You need this first |
|---|---|---|
| 1 | Train | venv + `pip install` |
| 2 | Tests | venv |
| 3 | MLflow UI | a finished training run (`mlflow.db`) |
| 4 | API on your machine | the ONNX file from training |
| 5 | Docker | the ONNX file |
| 6 | Kubernetes | the Docker image |
| 7 | Prometheus and Grafana | Kubernetes (they come from the same `kubectl apply`) |
| 8 | GitHub Actions | a push to GitHub |

---

## 1. Train the model

```bash
source venv/bin/activate
PYTHONPATH=src python -m mlops.main
```

That runs `src/mlops/main.py`. It loads `configs/config.yaml`, then:

- `DataLoader` downloads `data/raw/heart_disease.csv` if the file is missing (`fetch_ucirepo(id=45)`).
- `Preprocessor.clean` drops missing `ca` / `thal` and sets `target = 1` when the original value is greater than 0.
- `EDA` writes charts to `reports/figures/`.
- `ModelTrainer` splits 80/20 (stratified, `random_state=42`), tunes both models with 5-fold cross-validation, logs MLflow, and writes `models/heart_disease.onnx`.

When it finishes, the terminal prints each model’s best parameters and CV scores. The packaged model is whichever one has the higher `cv_roc_auc` (random forest on the run used for this assignment).

Files this creates:

- `data/raw/heart_disease.csv`
- `data/cleaned/heart_disease.csv`, `train.csv`, `test.csv`, `overview.csv`
- `reports/figures/*.png`
- `models/heart_disease.onnx`
- `mlflow.db` (SQLite database of experiments)

Training takes a few minutes because it fits many grid-search combinations.

## 2. Run the tests

```bash
source venv/bin/activate
PYTHONPATH=src pytest tests/ -q
```

Pytest runs `tests/`. A quiet green run means the checks passed. Lint (style check) is separate:

```bash
ruff check src tests
```

| Test file | What it checks |
|---|---|
| `tests/conftest.py` | Shared sample patient and fake data used by the other tests |
| `tests/test_data.py` | Cleaning drops missing `ca`/`thal`, target becomes 0/1, dummy encoding |
| `tests/test_model.py` | Both model names exist, feature groups are right, a tiny fit/predict works |
| `tests/test_predict.py` | The ONNX predictor returns a prediction and a probability |
| `tests/test_api.py` | `/`, `/health`, `POST /predict`, `/monitor`, `/metrics` |

## 3. Open MLflow

Leave training finished, then in the project folder:

```bash
source venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the URL it prints (usually `http://127.0.0.1:5000`). Experiment name: `heart-disease`.

You should see one parent run `compare-models` and two nested runs, `logistic_regression` and `random_forest`. Each run stores parameters, CV metrics, ROC / precision-recall / confusion plots, the CSVs, `config.yaml`, and `requirements.txt`. The parent run also has the ONNX file under the `onnx` artifact folder.

Stop the UI with Ctrl+C in that terminal. `mlflow.db` stays on disk.

## 4. Start the API on your machine

```bash
source venv/bin/activate
PYTHONPATH=src uvicorn mlops.api:app --host 0.0.0.0 --port 8000
```

Leave that terminal running. In a browser open `http://127.0.0.1:8000/`.

| URL | What it is |
|---|---|
| `GET /` | Form. Fill numbers and submit. The page shows JSON. |
| `GET /docs` | Interactive API docs generated by FastAPI |
| `GET /health` | `{"status": "ok", ...}` — Kubernetes uses this to see if the pod is alive |
| `GET /predict` | Tells you to POST, and lists the feature names |
| `POST /predict` | Send the 13 clinical fields as JSON. Returns `prediction` and `probability` |
| `GET /monitor` | Last 100 requests: method, path, status, latency |
| `GET /metrics` | Prometheus text. This path is not stored in the `/monitor` table |

Sample request (second terminal):

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

A healthy response looks like `{"prediction": 0, "probability": 0.336}`. The exact probability depends on the trained model.

Stop the API with Ctrl+C before you start Docker on the same port 8000, or the port will already be taken.

### How logging works

`src/mlops/api.py` has middleware that times every request. `src/mlops/monitor.py` (`RequestMonitor`) does three things with that timing:

1. Writes a line to the process log: `POST /predict -> 200 in 12.3ms`. You see this in the terminal that is running uvicorn, and later in `kubectl logs`.
2. Keeps the last 100 calls in memory for the `/monitor` page.
3. Increments Prometheus metrics:
   - `heart_api_requests_total{method, path, status}`
   - `heart_api_request_latency_seconds{method, path}`

`/metrics` itself is skipped so Prometheus scrapes do not flood the request log.

---

## 5. Docker

Docker packages the API so it runs the same way on any machine. The recipe is `Dockerfile`:

1. Start from `python:3.12-slim`.
2. `pip install -r requirements.txt`.
3. Copy `src/`, `configs/`, and `models/heart_disease.onnx`.
4. Start `uvicorn mlops.api:app` on port 8000.

Build and run (API on your machine must be stopped first):

```bash
docker build -t heart-api:local .
docker run --rm -p 8000:8000 heart-api:local
```

`-t heart-api:local` is the image name Kubernetes will look for. `-p 8000:8000` publishes container port 8000 on your laptop. `--rm` deletes the container when you stop it.

Open `http://127.0.0.1:8000/` again and send the same curl. Ctrl+C stops the container.

Useful checks:

```bash
docker images | grep heart-api
docker ps
```

---

## 6. Kubernetes (Minikube)

Kubernetes runs the container as pods and gives them a stable address. This project uses **Minikube**, a one-node cluster on your laptop, and **plain YAML** in `k8s/` (there is no Helm chart).

Start the cluster and load the image into it. Minikube has its own Docker, so an image you built on the host is invisible until you load it.

```bash
minikube start --driver=docker
minikube image load heart-api:local
kubectl apply -f k8s/
```

`kubectl apply -f k8s/` creates everything in that folder:

| File | What it creates |
|---|---|
| `k8s/deployment.yaml` | `heart-api` deployment: **2 copies** of image `heart-api:local`, port 8000, health checks on `/health`. `imagePullPolicy: Never` means “use the image already loaded into Minikube, do not download it from the internet.” |
| `k8s/service.yaml` | Service named `heart-api`, type LoadBalancer, port 8000. Traffic to the service is shared across the 2 pods. |
| `k8s/prometheus.yaml` | Prometheus plus its config. It scrapes `heart-api:8000/metrics` every 5 seconds. Its own service is a LoadBalancer on port 9090. |
| `k8s/grafana.yaml` | Grafana (user `admin`, password `admin`) with a Prometheus datasource and a dashboard called **Heart Disease API**. Service port 3000. |

Check that pods are running:

```bash
kubectl get pods
kubectl get svc
```

You want `heart-api` with `2/2` ready, plus one `prometheus` pod and one `grafana` pod, all `Running`.

Get a URL:

```bash
minikube service heart-api --url
```

That prints something like `http://192.168.49.2:3xxxx`. Open it in the browser. The port changes every time the service is recreated, so always use the command rather than an old URL from the report.

If the service has no external IP, run this in another terminal and leave it open (it asks for your password):

```bash
minikube tunnel
```

Pod logs (this is the same request logging as the local API):

```bash
kubectl logs -l app=heart-api --tail=50
```

Restart after you rebuild the image:

```bash
docker build -t heart-api:local .
minikube image load heart-api:local
kubectl rollout restart deployment/heart-api
```

Stop the cluster when you are done (your manifests stay in the repo; the running cluster does not):

```bash
minikube stop
```

Start it again later with `minikube start --driver=docker`. If the pods are gone, run `kubectl apply -f k8s/` again. You only need `minikube image load` again if you rebuilt the image.

---

## 7. Prometheus and Grafana

They start with the same `kubectl apply -f k8s/`. You do not install them separately.

```bash
minikube service prometheus --url
minikube service grafana --url
```

**Prometheus.** Open the URL, then Status → Targets. The job `heart-api` should say **UP**. In the query box try:

```text
heart_api_requests_total
```

**Grafana.** Log in with `admin` / `admin`. Open the dashboard **Heart Disease API**. It plots request rate and p95 latency. Submit the API form, or POST `/predict` a few times with curl. The health probes already call `/health`. Then refresh Grafana.

---

## 8. GitHub Actions

`.github/workflows/ci.yml` runs on every push and every pull request, on GitHub’s machines, not on your laptop. The job does this:

1. Check out the code.
2. Install Python 3.12 and `pip install -r requirements.txt`.
3. `ruff check src tests` — the job fails if lint fails.
4. `PYTHONPATH=src pytest tests/ -q` — the job fails if a test fails.
5. `PYTHONPATH=src python -m mlops.main` — trains again in CI.
6. Uploads `models/heart_disease.onnx`, `reports/figures`, and `mlflow.db` as a downloadable artifact even if a later step failed (`if: always()`).

To see it:

```bash
git add -A
git commit -m "Describe your change"
git push
```

On GitHub open the **Actions** tab. A green run is the screenshot the assignment asks for. You need a GitHub remote first (`git remote -v`). If there is no remote yet, create a repository on GitHub and follow the “push an existing repository” commands it shows.

---

## Every project file

Ignore `venv/`. That is installed packages, not your code. `mlruns/` and `mlflow.db` are created when you train. They are not part of the submission. GitHub Actions trains again and uploads a fresh `mlflow.db`.

### Code (`src/mlops/`)

| File | Job |
|---|---|
| `config.py` | Reads `configs/config.yaml` and turns relative paths into absolute paths from the repo root, so the same YAML works no matter which folder you launched from. |
| `data_loader.py` | Downloads the UCI dataset if `data/raw/heart_disease.csv` is missing, then loads it. |
| `preprocessing.py` | `clean`: drop missing `ca`/`thal`, cast them to int, binarize `target`. `encode` is the pandas dummy-encoding helper used by tests. The model’s real encoding is inside the sklearn pipeline in `train.py`, so the test set never sees training-set statistics. |
| `eda.py` | Saves class-balance, histogram, correlation, and train-vs-test plots under `reports/figures/`. |
| `train.py` | Splits data, grid-searches both models, logs MLflow, exports the winner to ONNX. |
| `main.py` | The training entry point: load → clean → save CSV → EDA → train. |
| `predict.py` | Loads `models/heart_disease.onnx` with ONNX Runtime and returns prediction plus probability. |
| `api.py` | FastAPI app: form, `/predict`, `/health`, `/monitor`, `/metrics`, and the request middleware. |
| `monitor.py` | In-memory request log plus Prometheus counter and histogram. |
| `__init__.py` | Marks the folder as a Python package. |

### Config, data, model

| Path | Job |
|---|---|
| `configs/config.yaml` | Dataset id, file paths, which columns are numeric / categorical / binary, the hyperparameter grids, MLflow experiment name, ONNX path, API host and port. Change behavior here before changing code. |
| `requirements.txt` | Pinned Python packages for training, tests, and the Docker image. |
| `data/raw/` | Original download. |
| `data/cleaned/` | Cleaned table plus the train and test splits. |
| `models/heart_disease.onnx` | The model the API actually loads. If this file is missing, train first. |

### Tests, container, cluster, CI

| Path | Job |
|---|---|
| `tests/` | Pytest suite described above. |
| `Dockerfile` | Image build for the API. |
| `.dockerignore` | Keeps `venv`, git, reports, and tests out of the image so the build stays smaller. |
| `k8s/*.yaml` | Kubernetes manifests described above. |
| `.github/workflows/ci.yml` | GitHub Actions pipeline. |
| `.gitignore` | Tells git to skip `venv/`, caches, `mlruns/`, and `mlflow.db`. |

### Write-up and pictures

| Path | Job |
|---|---|
| `README.md` | Short commands. |
| `GUIDE.md` | This file. |
| `reports/MLOps_Assignment1_Report.md` | Assignment write-up in Markdown. |
| `reports/MLOps_Assignment1_Report.docx` | Same write-up for submission. |
| `reports/figures/` | Charts produced by training. |
| `reports/video/MLOps_Assignment1_Demo.mp4` | Demo recording. |
| `screenshots/` | Pictures pasted into the report (API, MLflow, pods, Prometheus, Grafana). |
| `assignment/` | The original assignment instructions (HTML export of the PDF). |

---

## If something fails

**`ModuleNotFoundError: mlops`.** You forgot `PYTHONPATH=src`, or the venv is not activated.

**`/predict` returns 500 and the log says the ONNX file is missing.** Run training first so `models/heart_disease.onnx` exists, then rebuild the Docker image if you are using Docker.

**Port 8000 already in use.** Another uvicorn or `docker run` is still up. Stop it with Ctrl+C, or `docker ps` and `docker stop <id>`.

**Kubernetes `ImagePullBackOff` or `ErrImageNeverPull`.** The cluster does not have `heart-api:local`. Run `minikube image load heart-api:local` again, then `kubectl rollout restart deployment/heart-api`.

**Pods `CrashLoopBackOff`.** Read the reason: `kubectl logs -l app=heart-api --tail=80`.

**Grafana charts are flat.** Prometheus is up, but nobody has called `/predict`. Submit the form or send the sample `curl` from the README.

**MLflow UI is empty.** Start it from the project folder with `sqlite:///mlflow.db`, after at least one training run. Do not start it from inside `src/`.
