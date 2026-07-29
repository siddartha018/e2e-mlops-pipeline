"""Client interfaces for P2 (inference), P4 (vector store), P5 (task queue).

Every client has an abstract interface with two implementations:
- Local*  : in-process implementation that works today
- HttpP*  : HTTP client that will connect to the from-scratch service
"""
from .inference import InferenceClient, LocalInferenceClient, HttpP2InferenceClient
from .task_queue import TaskQueueClient, LocalRunner, HttpP5Client, JobSpec, JobStatus
from .vector_store import VectorStoreClient, FaissLocalStore, HttpP4Client

__all__ = [
    "InferenceClient",
    "LocalInferenceClient",
    "HttpP2InferenceClient",
    "TaskQueueClient",
    "LocalRunner",
    "HttpP5Client",
    "JobSpec",
    "JobStatus",
    "VectorStoreClient",
    "FaissLocalStore",
    "HttpP4Client",
]
