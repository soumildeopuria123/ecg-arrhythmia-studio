"""Database interface for ECG Arrhythmia Classification Studio.

Provides async-compatible collection operations for jobs and predictions.
Uses Motor (MongoDB) pattern if available, falls back to mock implementations.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try to import Motor for real MongoDB, fall back to mock
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    _motor_available = True
except ImportError:
    _motor_available = False
    logger.info("Motor not available, using mock database interface")

# Collection placeholders - initialized on first use
jobs_collection = None
predictions_collection = None


def init_db(db_client=None):
    """Initialize database collections.
    
    Parameters
    ----------
    db_client : Optional[AsyncIOMotorClient]
        MongoDB client instance. If None, uses mock collections.
    """
    global jobs_collection, predictions_collection
    
    if _motor_available and db_client is not None:
        # Real MongoDB collections
        jobs_collection = db_client.ecg_studio.jobs
        predictions_collection = db_client.ecg_studio.predictions
        logger.info("MongoDB collections initialized")
    else:
        # Mock collections using simple dict-based storage
        # In production, replace with real MongoDB or PostgreSQL
        if jobs_collection is None:
            jobs_collection = _MockCollection("jobs")
        if predictions_collection is None:
            predictions_collection = _MockCollection("predictions")
        logger.info("Mock database collections initialized")


class _MockCollection:
    """Mock collection for development/testing without MongoDB."""
    
    def __init__(self, name: str):
        self.name = name
        self._data: Dict[str, Dict[str, Any]] = {}
        self._id_counter = 0
    
    async def insert_one(self, document: Dict[str, Any]) -> None:
        """Insert a single document."""
        doc_id = document.get("id", document.get("job_id", document.get("prediction_id", "")))
        self._data[doc_id] = {
            **document,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        logger.debug(f"Mock inserted into {self.name}: {doc_id}")
    
    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find single document matching query."""
        for doc_id, doc in self._data.items():
            if all(
                doc.get(k) == v for k, v in query.items()
                if k in doc
            ):
                return doc
        return None
    
    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Update single document matching query."""
        doc = await self.find_one(query)
        if doc:
            for key, value in update.items():
                if key.startswith("$"):
                    # Handle MongoDB-style operators
                    if key == "$set":
                        for k, v in value.items():
                            doc[k] = v
                else:
                    doc[key] = value
            doc["updated_at"] = datetime.now(timezone.utc)
            logger.debug(f"Mock updated in {self.name}")
    
    async def delete_one(self, query: Dict[str, Any]) -> None:
        """Delete single document matching query."""
        doc_id = None
        for did, doc in self._data.items():
            if all(doc.get(k) == v for k, v in query.items() if k in doc):
                doc_id = did
                break
        if doc_id and doc_id in self._data:
            del self._data[doc_id]
            logger.debug(f"Mock deleted from {self.name}: {doc_id}")