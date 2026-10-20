import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from mlops.config import load_config
from mlops.monitor import CONTENT_TYPE, RequestMonitor
from mlops.predict import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

config = load_config()
predictor = Predictor(config)
monitor = RequestMonitor()
app = FastAPI(title="Heart Disease API", version="1.0.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path != "/metrics":
        monitor.record(
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - start,
        )
    return response


class HeartFeatures(BaseModel):
    age: float
    sex: int = Field(ge=0, le=1)
    cp: int
    trestbps: float
    chol: float
    fbs: int = Field(ge=0, le=1)
    restecg: int
    thalach: float
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Heart Disease API</title>
    <style>
      body {{ font-family: sans-serif; max-width: 720px; margin: 2rem auto; }}
      label {{ display: block; margin: 0.4rem 0; }}
      input {{ width: 8rem; }}
      pre {{ background: #f4f4f4; padding: 0.75rem; }}
    </style>
  </head>
  <body>
    <h1>Heart Disease API</h1>
    <p>Scores the packaged ONNX model at <code>{predictor.model_path.name}</code>.</p>
    <p>
      <a href="/docs">OpenAPI docs</a> ·
      <a href="/health">Health</a> ·
      <a href="/predict">Predict</a> ·
      <a href="/monitor">Monitor</a> ·
      <a href="/metrics">Metrics</a>
    </p>
    <form id="form">
      <label>age <input name="age" type="number" step="any" value="63" /></label>
      <label>sex <input name="sex" type="number" value="1" /></label>
      <label>cp <input name="cp" type="number" value="1" /></label>
      <label>trestbps <input name="trestbps" type="number" step="any" value="145" /></label>
      <label>chol <input name="chol" type="number" step="any" value="233" /></label>
      <label>fbs <input name="fbs" type="number" value="1" /></label>
      <label>restecg <input name="restecg" type="number" value="2" /></label>
      <label>thalach <input name="thalach" type="number" step="any" value="150" /></label>
      <label>exang <input name="exang" type="number" value="0" /></label>
      <label>oldpeak <input name="oldpeak" type="number" step="any" value="2.3" /></label>
      <label>slope <input name="slope" type="number" value="3" /></label>
      <label>ca <input name="ca" type="number" value="0" /></label>
      <label>thal <input name="thal" type="number" value="6" /></label>
      <button type="submit">Predict</button>
    </form>
    <pre id="result">Submit a patient row to get a prediction.</pre>
    <script>
      document.getElementById("form").addEventListener("submit", async (event) => {{
        event.preventDefault();
        const data = Object.fromEntries(new FormData(event.target));
        for (const key of Object.keys(data)) data[key] = Number(data[key]);
        const response = await fetch("/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(data),
        }});
        document.getElementById("result").textContent = JSON.stringify(await response.json(), null, 2);
      }});
    </script>
  </body>
</html>
"""


@app.get("/metrics")
def metrics() -> Response:
    return Response(monitor.export(), media_type=CONTENT_TYPE)


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page() -> str:
    rows = "".join(
        f"<tr><td>{item['method']}</td><td>{item['path']}</td>"
        f"<td>{item['status']}</td><td>{item['latency_ms']}</td></tr>"
        for item in monitor.snapshot()
    ) or "<tr><td colspan='4'>No requests yet.</td></tr>"
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>API monitor</title>
    <style>
      body {{ font-family: sans-serif; max-width: 800px; margin: 2rem auto; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 0.4rem; text-align: left; }}
    </style>
  </head>
  <body>
    <h1>API request log</h1>
    <p><a href="/">Home</a> · <a href="/metrics">Prometheus metrics</a></p>
    <table>
      <tr><th>Method</th><th>Path</th><th>Status</th><th>Latency (ms)</th></tr>
      {rows}
    </table>
  </body>
</html>
"""


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_path": str(predictor.model_path),
    }


@app.get("/predict")
def predict_info() -> dict:
    return {
        "method": "POST",
        "features": predictor.columns,
    }


@app.post("/predict")
def predict(features: HeartFeatures) -> dict:
    try:
        return predictor.predict(features.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mlops.api:app", host=config.api.host, port=int(config.api.port))
