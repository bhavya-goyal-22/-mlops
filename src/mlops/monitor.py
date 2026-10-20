import logging
from collections import deque

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

logger = logging.getLogger("mlops.api")


class RequestMonitor:
    def __init__(self, maxlen: int = 100):
        self.recent = deque(maxlen=maxlen)
        self.requests = Counter(
            "heart_api_requests_total",
            "API requests",
            ["method", "path", "status"],
        )
        self.latency = Histogram(
            "heart_api_request_latency_seconds",
            "API request latency",
            ["method", "path"],
        )

    def record(self, method: str, path: str, status: int, seconds: float) -> None:
        path = path.split("?", 1)[0]
        self.requests.labels(method, path, str(status)).inc()
        self.latency.labels(method, path).observe(seconds)
        entry = {
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": round(seconds * 1000, 2),
        }
        self.recent.appendleft(entry)
        logger.info("%s %s -> %s in %.1fms", method, path, status, seconds * 1000)

    def snapshot(self) -> list[dict]:
        return list(self.recent)

    def export(self) -> bytes:
        return generate_latest()


CONTENT_TYPE = CONTENT_TYPE_LATEST
