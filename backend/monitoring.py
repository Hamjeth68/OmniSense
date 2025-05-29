# File: omniverse_ai/monitoring.py
"""
Monitoring and logging setup for OmniSense AI
"""

import logging
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time

# Prometheus metrics
REQUEST_COUNTER = Counter(
    'omniverse_requests_total',
    'Total number of requests processed',
    ['modality', 'status']
)

PROCESSING_TIME = Histogram(
    'omniverse_processing_time_seconds',
    'Time spent processing requests',
    ['modality']
)

MODEL_LOAD_TIME = Gauge(
    'omniverse_model_load_time_seconds',
    'Time taken to load models'
)

# Logging configuration
def configure_logging():
    """Configure structured logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('omniverse.log')
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

class PerformanceMonitor:
    """Performance monitoring decorator"""
    def __init__(self, metric, label=None):
        self.metric = metric
        self.label = label
    
    def __call__(self, func):
        def wrapped(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if self.label:
                    self.metric.labels(self.label).observe(duration)
                else:
                    self.metric.observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                if self.label:
                    self.metric.labels(self.label).observe(duration)
                else:
                    self.metric.observe(duration)
                raise e
        return wrapped

def start_monitoring(port=8001):
    """Start Prometheus metrics server"""
    start_http_server(port)
    logging.info(f"Monitoring server started on port {port}")