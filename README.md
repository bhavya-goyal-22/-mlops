# Heart Disease MLOps Platform

End-to-end classifier for [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (id 45): download, clean, EDA, two models, MLflow, ONNX, FastAPI, Docker, Minikube, Prometheus, Grafana.

For what each file does and how to start training, the API, Docker, Kubernetes, logging, Prometheus, Grafana, and GitHub Actions, read [GUIDE.md](GUIDE.md).

Paths, columns, and hyperparameters live in `configs/config.yaml`. Classes in `src/mlops/` own the behavior.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
PYTHONPATH=src python -m mlops.main
PYTHONPATH=src pytest tests/ -q
```

Training writes cleaned CSVs to `data/cleaned/`, figures to `reports/figures/`, the packaged model to `models/heart_disease.onnx`, and the MLflow experiment `heart-disease` to `mlflow.db`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Experiment name: `heart-disease`.

## Serve

```bash
PYTHONPATH=src uvicorn mlops.api:app --host 0.0.0.0 --port 8000
```

Stop this process before `docker run` on the same port.

```bash
docker build -t heart-api:local .
docker run --rm -p 8000:8000 heart-api:local
```

Local API and the Docker container use the same URLs:

| Page | URL |
|---|---|
| Form | [http://127.0.0.1:8000/](http://127.0.0.1:8000/) |
| OpenAPI | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| Health | [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) |
| Predict contract (GET) | [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) |
| Request log | [http://127.0.0.1:8000/monitor](http://127.0.0.1:8000/monitor) |
| Prometheus text | [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics) |

`POST /predict` returns `{ "prediction": 0|1, "probability": … }`:

```bash
curl -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}' \
  http://127.0.0.1:8000/predict
```

## Deployed API URL (if public) or access instructions (for local testing)

There is no public URL. The deployed API is local Minikube (Docker driver): two replicas, LoadBalancer on port 8000. Manifests are in `k8s/`.

```bash
minikube start --driver=docker
docker build -t heart-api:local .
minikube image load heart-api:local
kubectl apply -f k8s/
minikube service heart-api --url
minikube service prometheus --url
minikube service grafana --url
```

If a page does not open, leave `minikube tunnel` running in another terminal and print the three URLs again. The host is the Minikube IP. The port is the service NodePort and changes when the service is recreated, so prefer the command output over a saved address.

On the cluster used for the screenshots the addresses were:

| What to open | URL |
|---|---|
| API form | [http://192.168.49.2:31694/](http://192.168.49.2:31694/) |
| Health | [http://192.168.49.2:31694/health](http://192.168.49.2:31694/health) |
| Predict contract (GET) | [http://192.168.49.2:31694/predict](http://192.168.49.2:31694/predict) |
| Request log | [http://192.168.49.2:31694/monitor](http://192.168.49.2:31694/monitor) |
| API metrics | [http://192.168.49.2:31694/metrics](http://192.168.49.2:31694/metrics) |
| Prometheus, then Status → Targets, job `heart-api` | [http://192.168.49.2:30275/](http://192.168.49.2:30275/) |
| Prometheus targets | [http://192.168.49.2:30275/targets](http://192.168.49.2:30275/targets) |
| Grafana login `admin` / `admin`, dashboard **Heart Disease API** | [http://192.168.49.2:31957/](http://192.168.49.2:31957/) |

Submit the form on the API URL, or:

```bash
URL="$(minikube service heart-api --url)"
curl "$URL/health"
curl -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}' "$URL/predict"
```

Submit the form, or repeat the `curl` above, so the Grafana charts are not empty. The health probes already call `/health`.

Pod log lines (`POST /predict -> 200 in 13.2ms`):

```bash
kubectl logs -l app=heart-api --tail=50
```

## Repository

| Path | Role |
|---|---|
| `src/mlops/` | pipeline, trainer, ONNX predictor, API |
| `configs/config.yaml` | dataset id, columns, models, ports |
| `tests/` | data, models, predictor, API |
| `models/heart_disease.onnx` | packaged Random Forest |
| `data/` | raw + cleaned CSVs |
| `Dockerfile` + `requirements.txt` | serving image (same pinned file as training) |
| `k8s/` | API (2 replicas, LoadBalancer), Prometheus, Grafana |
| `.github/workflows/ci.yml` | lint, test, train |
| `screenshots/` | assignment screenshots |
| `reports/` | figures, report, demo video |
