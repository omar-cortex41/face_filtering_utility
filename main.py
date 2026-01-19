"""
Visitor Face Clustering Utility - Main Application
Combines database operations, clustering, and web API in one file.
"""
import psycopg2
import json
from pathlib import Path
import yaml
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import cv2
import pickle
import time
import os
import threading
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn


# ============== Background Task Status ==============
@dataclass
class TaskStatus:
    """Track background task progress."""
    running: bool = False
    task_name: str = ""
    progress: int = 0  # 0-100
    message: str = ""
    error: Optional[str] = None

_task_status = TaskStatus()

# MediaPipe for frontal face detection
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp_face_detection = None
    MEDIAPIPE_AVAILABLE = False

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DB_CONFIG = config["database"]
QDRANT_URL = config["qdrant"]["url"]
QDRANT_COLLECTION = config["qdrant"]["collection"]

# Photo configuration - directory name is "photos" which matches DB paths and URL prefix
PHOTOS_DIR = Path(config["photos"]["directory"])  # e.g., "photos"
PHOTOS_PREFIX = config["photos"]["directory"]      # Same as dir name, used for URLs

FACE_FILTER_CONFIG = config.get("face_filter", {})
CLUSTERING_CONFIG = config.get("clustering", {})


def get_db_connection():
    """Create and return a PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_qdrant_client():
    """Create and return a Qdrant client."""
    return QdrantClient(url=QDRANT_URL)


def parse_attributes(attrs):
    """Parse object_attributes from string or dict."""
    if isinstance(attrs, str):
        try:
            return json.loads(attrs)
        except json.JSONDecodeError:
            return None
    return attrs if isinstance(attrs, dict) else None


def get_person_name_by_object_id(object_id: int) -> str | None:
    """
    Get the name of a person given the object_id (same as Qdrant point ID).
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT object_attributes FROM objects WHERE object_id = %s",
                (object_id,)
            )
            row = cur.fetchone()
            if not row:
                return None

            attrs = parse_attributes(row[0])
            if attrs:
                return attrs.get("name") or attrs.get("person_name")
            return None


def get_qdrant_point(point_id: int):
    """Retrieve a single point from Qdrant by ID (fast lookup)."""
    client = get_qdrant_client()
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    return points[0] if points else None


def get_visitors() -> dict[str, int]:
    """
    Get all registered visitors with their object_id.
    Returns {name: object_id}.

    Supports two data formats:
    1. Objects with visitor_id attribute (legacy)
    2. Objects linked via fr_logs with is_registered=true (production)
    """
    visitors = {}

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # First try: legacy format with visitor_id
            cur.execute("""
                SELECT object_attributes->>'name' as name, object_id
                FROM objects
                WHERE object_attributes ? 'visitor_id'
            """)
            for row in cur.fetchall():
                if row[0]:
                    visitors[row[0]] = row[1]

            # Second try: production format via fr_logs
            if not visitors:
                cur.execute("""
                    SELECT DISTINCT o.object_id, o.object_attributes->>'image_path' as img_path
                    FROM objects o
                    JOIN fr_logs f ON o.object_id = f.object_id
                    WHERE f.is_registered = true
                    AND o.object_attributes->>'image_path' IS NOT NULL
                """)
                for row in cur.fetchall():
                    object_id = row[0]
                    img_path = row[1]
                    # Extract name from image_path: photos/Name_ID/image.jpg -> Name
                    if img_path:
                        parts = img_path.split('/')
                        if len(parts) >= 2:
                            folder_name = parts[1]  # e.g., "Najada Gjonaj_J259223044Q"
                            name = folder_name.rsplit('_', 1)[0]  # "Najada Gjonaj"
                            visitors[name] = object_id

    return visitors


def find_similar_embeddings(object_id: int, similarity_threshold: float = 0.2, limit: int = 1000) -> list[int]:
    """
    Find all Qdrant point IDs similar to the given object_id's embedding.
    (Legacy cosine similarity method - kept for backwards compatibility)
    """
    client = get_qdrant_client()

    # Get the embedding for this object_id
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[object_id],
        with_vectors=True,
    )

    if not points or not points[0].vector:
        return []

    vector = points[0].vector

    # Search for all similar embeddings
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=limit,
        score_threshold=similarity_threshold,
    )

    return [hit.id for hit in result.points]


# ========= Frontal Face Filter =========
def is_frontal_face(image_path: str) -> bool:
    """
    Check if the face in the image is frontal using MediaPipe.
    Returns True if frontal (or if filter is disabled/unavailable), False otherwise.
    """
    # Check if filter is enabled
    if not FACE_FILTER_CONFIG.get("enabled", True):
        return True

    if not MEDIAPIPE_AVAILABLE:
        print("[WARN] MediaPipe not available, skipping frontal face filter")
        return True

    # Get thresholds from config
    yaw_max = float(FACE_FILTER_CONFIG.get("yaw_max", 0.25))
    yaw_min = float(FACE_FILTER_CONFIG.get("yaw_min", -0.25))
    blur_min = float(FACE_FILTER_CONFIG.get("blur_min", 5.0))
    min_confidence = float(FACE_FILTER_CONFIG.get("min_detection_confidence", 0.3))

    # Load image
    try:
        # Handle path - remove leading slash if present
        img_path = image_path.lstrip("/")
        if not Path(img_path).exists():
            return False

        frame = cv2.imread(img_path)
        if frame is None:
            return False

        H, W = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_val < blur_min:
            return False

        # Detect face
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence
        ) as detector:
            results = detector.process(rgb_frame)

        if not results.detections:
            return False

        # Get largest detection
        def area_of(d):
            bb = d.location_data.relative_bounding_box
            return max(bb.width, 0) * max(bb.height, 0)

        detection = max(results.detections, key=area_of)
        keypoints = detection.location_data.relative_keypoints
        # Keypoints: [R_eye, L_eye, nose, mouth, R_ear, L_ear]

        if len(keypoints) < 6:
            return False

        r_eye, l_eye, nose, mouth, r_ear, l_ear = keypoints[:6]

        # Calculate yaw from ear-to-nose distances
        dx_r = abs(nose.x - r_ear.x)
        dx_l = abs(l_ear.x - nose.x)
        denom = max(dx_r + dx_l, 1e-6)
        yaw = (dx_l - dx_r) / denom  # +right, -left

        # Check if frontal (yaw within thresholds)
        is_frontal = yaw_min <= yaw <= yaw_max

        return is_frontal

    except Exception as e:
        print(f"[WARN] Frontal face check failed for {image_path}: {e}")
        return True  # Default to True on error to not block


# Cache for frontal face check results (persisted in DB)
_frontal_cache: dict[int, bool] = {}
_frontal_cache_loaded = False


def _load_frontal_cache():
    """Load frontal check cache from DB."""
    global _frontal_cache, _frontal_cache_loaded
    if _frontal_cache_loaded:
        return

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT object_id,
                           (object_attributes->>'is_frontal')::boolean
                    FROM objects
                    WHERE object_attributes->>'is_frontal' IS NOT NULL
                """)
                for row in cur.fetchall():
                    _frontal_cache[row[0]] = row[1]
        _frontal_cache_loaded = True
        print(f"[Frontal] Loaded {len(_frontal_cache)} cached results from DB")
    except Exception as e:
        print(f"[Frontal] Failed to load cache: {e}")


def _save_frontal_to_db(object_id: int, is_frontal: bool):
    """Save frontal check result to DB for caching."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE objects
                    SET object_attributes = object_attributes || %s::jsonb
                    WHERE object_id = %s
                """, (json.dumps({"is_frontal": is_frontal}), object_id))
                conn.commit()
    except Exception as e:
        pass  # Silently fail - cache is optional


def _resolve_photo_path(image_path: str) -> str:
    """
    Resolve the actual file path for an image.
    DB stores paths like 'photos/unknown/...' matching the photos directory.
    """
    if not image_path:
        return image_path

    # Strip leading slash
    path = image_path.lstrip("/")

    # Path should exist as-is since DB path matches directory structure
    if Path(path).exists():
        return path

    # Return original path (will fail gracefully in is_frontal_face)
    return path


def is_frontal_cached(object_id: int, image_path: str) -> bool:
    """Check if face is frontal, using cache first."""
    global _frontal_cache
    _load_frontal_cache()

    # Check cache first
    if object_id in _frontal_cache:
        return _frontal_cache[object_id]

    # Resolve actual path (handle photos/ vs albania_photos/ mismatch)
    resolved_path = _resolve_photo_path(image_path)

    # Run actual check
    result = is_frontal_face(resolved_path)

    # Cache result
    _frontal_cache[object_id] = result
    _save_frontal_to_db(object_id, result)

    return result


def filter_frontal_faces(object_ids: list[int], get_image_path_func) -> list[int]:
    """
    Filter a list of object IDs to only include those with frontal faces.

    Args:
        object_ids: List of object IDs to filter
        get_image_path_func: Function to get image path from object ID

    Returns:
        Filtered list of object IDs with frontal faces
    """
    if not FACE_FILTER_CONFIG.get("enabled", True):
        return object_ids

    frontal_ids = []
    for oid in object_ids:
        img_path = get_image_path_func(oid)
        if img_path and is_frontal_face(img_path):
            frontal_ids.append(oid)
        elif not img_path:
            # Include IDs without images (can't verify)
            frontal_ids.append(oid)

    return frontal_ids


# Cache for clustering results
_cluster_cache = {
    "labels": None,
    "point_ids": None,
    "n_clusters": None,
}


def get_all_embeddings(update_progress: bool = False) -> tuple[list[int], np.ndarray]:
    """Retrieve all embeddings from Qdrant with progress tracking."""
    global _task_status
    client = get_qdrant_client()

    # Get total count first for progress
    collection_info = client.get_collection(QDRANT_COLLECTION)
    total_points = collection_info.points_count

    # Scroll through all points with larger batch size
    all_points = []
    offset = None
    batch_size = 5000  # Much larger batches = fewer round trips

    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
        )
        all_points.extend(points)

        # Update progress (0-30% for loading embeddings)
        if update_progress and total_points > 0:
            pct = min(30, int((len(all_points) / total_points) * 30))
            _task_status.progress = pct
            _task_status.message = f"Loading embeddings... {len(all_points):,}/{total_points:,}"

        if offset is None:
            break

    point_ids = [p.id for p in all_points]
    vectors = np.array([p.vector for p in all_points])

    return point_ids, vectors


# Disk cache for persistence across restarts
CLUSTER_CACHE_FILE = Path(__file__).parent / ".cluster_cache.pkl"


def _estimate_k(n_samples: int, n_visitors: int) -> int:
    """
    Fast K estimation using rule of thumb.
    Much faster than silhouette search.
    """
    # Rule of thumb: sqrt(n/2) is often a good starting point
    rule_of_thumb = int(np.sqrt(n_samples / 2))

    # At least as many clusters as known visitors + some for unknowns
    min_k = max(n_visitors + 10, 20)

    # Cap at reasonable maximum
    max_k = min(n_samples // 5, 300)

    return max(min_k, min(rule_of_thumb, max_k))


def _load_cache_from_disk() -> bool:
    """Load cluster cache from disk if available."""
    global _cluster_cache
    try:
        if CLUSTER_CACHE_FILE.exists():
            with open(CLUSTER_CACHE_FILE, "rb") as f:
                cached = pickle.load(f)
                # Validate cache has expected structure
                if all(k in cached for k in ["labels", "point_ids", "n_clusters", "timestamp"]):
                    _cluster_cache = cached
                    age = time.time() - cached.get("timestamp", 0)
                    print(f"[Cluster] Loaded cache from disk ({len(cached['point_ids'])} points, {age:.0f}s old)")
                    return True
    except Exception as e:
        print(f"[Cluster] Failed to load disk cache: {e}")
    return False


def _save_cache_to_disk():
    """Save cluster cache to disk for persistence."""
    global _cluster_cache
    try:
        cache_to_save = {**_cluster_cache, "timestamp": time.time()}
        with open(CLUSTER_CACHE_FILE, "wb") as f:
            pickle.dump(cache_to_save, f)
    except Exception as e:
        print(f"[Cluster] Failed to save cache: {e}")


def cluster_embeddings(force_refresh: bool = False, background: bool = False) -> dict[int, int]:
    """
    Cluster all embeddings using MiniBatchKMeans (fast).
    Returns: {point_id: cluster_label}

    If background=True, returns cached results immediately and runs refresh in background.
    """
    global _cluster_cache

    # Try memory cache first
    if not force_refresh and _cluster_cache["labels"] is not None:
        return dict(zip(_cluster_cache["point_ids"], _cluster_cache["labels"]))

    # Try disk cache (survives server restarts)
    if not force_refresh and _load_cache_from_disk():
        return dict(zip(_cluster_cache["point_ids"], _cluster_cache["labels"]))

    # If background requested and we have some cache, return it and refresh in background
    if background and _cluster_cache["labels"] is not None:
        threading.Thread(target=_run_clustering_background, daemon=True).start()
        return dict(zip(_cluster_cache["point_ids"], _cluster_cache["labels"]))

    # No cache - must run synchronously
    return _run_clustering_sync()


def _run_clustering_sync() -> dict[int, int]:
    """Run clustering synchronously (blocking)."""
    global _cluster_cache
    start_time = time.time()

    point_ids, vectors = get_all_embeddings(update_progress=False)

    if len(vectors) < 2:
        return {pid: 0 for pid in point_ids}

    visitors = get_visitors()
    optimal_k = _estimate_k(len(vectors), len(visitors))

    kmeans = MiniBatchKMeans(
        n_clusters=optimal_k,
        random_state=42,
        batch_size=min(2048, len(vectors)),
        n_init=1,  # Single init for speed
        max_iter=50,
    )
    labels = kmeans.fit_predict(vectors)

    _cluster_cache["labels"] = labels
    _cluster_cache["point_ids"] = point_ids
    _cluster_cache["n_clusters"] = optimal_k
    _save_cache_to_disk()

    elapsed = time.time() - start_time
    print(f"[Cluster] {optimal_k} clusters from {len(vectors)} embeddings in {elapsed:.2f}s")

    return dict(zip(point_ids, labels))


def _run_clustering_background():
    """Run clustering in background thread with progress updates."""
    global _cluster_cache, _task_status

    if _task_status.running:
        return  # Already running

    try:
        _task_status.running = True
        _task_status.task_name = "clustering"
        _task_status.progress = 0
        _task_status.message = "Starting clustering..."
        _task_status.error = None

        start_time = time.time()

        # Step 1: Load embeddings (0-30%)
        _task_status.message = "Loading embeddings from Qdrant..."
        point_ids, vectors = get_all_embeddings(update_progress=True)

        if len(vectors) < 2:
            _cluster_cache["labels"] = np.array([0] * len(point_ids))
            _cluster_cache["point_ids"] = point_ids
            _cluster_cache["n_clusters"] = 1
            _task_status.progress = 100
            _task_status.message = "Done (too few embeddings)"
            return

        # Step 2: Estimate K (30-35%)
        _task_status.progress = 35
        _task_status.message = "Estimating optimal cluster count..."
        visitors = get_visitors()
        optimal_k = _estimate_k(len(vectors), len(visitors))

        # Step 3: Run K-Means (35-95%)
        _task_status.progress = 40
        _task_status.message = f"Clustering {len(vectors):,} embeddings into {optimal_k} clusters..."

        kmeans = MiniBatchKMeans(
            n_clusters=optimal_k,
            random_state=42,
            batch_size=min(2048, len(vectors)),
            n_init=1,
            max_iter=50,
        )

        # Fit in chunks to update progress
        n_samples = len(vectors)
        chunk_size = max(1000, n_samples // 10)

        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            kmeans.partial_fit(vectors[i:end_idx])
            pct = 40 + int((end_idx / n_samples) * 50)
            _task_status.progress = pct
            _task_status.message = f"Clustering... {end_idx:,}/{n_samples:,}"

        # Final predict
        _task_status.progress = 92
        _task_status.message = "Assigning cluster labels..."
        labels = kmeans.predict(vectors)

        # Step 4: Save (95-100%)
        _task_status.progress = 95
        _task_status.message = "Saving results..."

        _cluster_cache["labels"] = labels
        _cluster_cache["point_ids"] = point_ids
        _cluster_cache["n_clusters"] = optimal_k
        _save_cache_to_disk()

        elapsed = time.time() - start_time
        _task_status.progress = 100
        _task_status.message = f"Done! {optimal_k} clusters in {elapsed:.1f}s"
        print(f"[Cluster] Background: {optimal_k} clusters from {len(vectors)} embeddings in {elapsed:.2f}s")

    except Exception as e:
        _task_status.error = str(e)
        _task_status.message = f"Error: {e}"
        print(f"[Cluster] Background error: {e}")
    finally:
        _task_status.running = False


def start_background_clustering():
    """Start clustering in background if not already running."""
    global _task_status
    if _task_status.running:
        return False
    threading.Thread(target=_run_clustering_background, daemon=True).start()
    return True


def get_task_status() -> dict:
    """Get current background task status."""
    return {
        "running": _task_status.running,
        "task_name": _task_status.task_name,
        "progress": _task_status.progress,
        "message": _task_status.message,
        "error": _task_status.error,
    }


def get_cluster_for_point(object_id: int) -> list[int]:
    """
    Get all point IDs in the same cluster as the given object_id.
    """
    clusters = cluster_embeddings()

    if object_id not in clusters:
        return []

    target_cluster = clusters[object_id]
    return [pid for pid, cluster in clusters.items() if cluster == target_cluster]


def get_cluster_info() -> dict:
    """Get clustering statistics."""
    cluster_embeddings()  # Ensure clustering is done
    return {
        "n_clusters": _cluster_cache["n_clusters"],
        "total_points": len(_cluster_cache["point_ids"]) if _cluster_cache["point_ids"] else 0,
    }


def analyze_user_embeddings(similarity_threshold: float = 0.2):
    """
    For each user, find all similar embeddings in Qdrant.
    """
    visitors = get_visitors()

    print(f"{'Name':<30} {'Object ID':<12} {'Similar Points':<15}")
    print("-" * 60)

    for name, object_id in visitors.items():
        similar_ids = find_similar_embeddings(object_id, similarity_threshold)
        print(f"{name:<30} {object_id:<12} {len(similar_ids):<15}")


def debug_point_search(point_id: int):
    """Debug: search for similar points to a specific point."""
    client = get_qdrant_client()

    # Get collection info
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"Collection: {QDRANT_COLLECTION}")
    print(f"Total points: {info.points_count}")
    print(f"Distance: {info.config.params.vectors.distance}")
    print()

    # Get the point
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[point_id],
        with_vectors=True,
        with_payload=True,
    )

    if not points:
        print(f"Point {point_id} not found!")
        return

    print(f"Point {point_id} payload: {points[0].payload}")
    print(f"Vector length: {len(points[0].vector) if points[0].vector else 'None'}")
    print()

    # Try different thresholds
    for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.0]:
        result = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=points[0].vector,
            limit=100,
            score_threshold=threshold,
        )
        print(f"Threshold {threshold}: {len(result.points)} similar points")
        if threshold == 0.0 and result.points:
            print(f"  Top 5 scores: {[round(p.score, 3) for p in result.points[:5]]}")


def get_embedding_for_object(object_id: int) -> np.ndarray | None:
    """Get the embedding vector for a specific object_id from Qdrant."""
    client = get_qdrant_client()
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[object_id],
        with_vectors=True,
    )
    if points and points[0].vector:
        return np.array(points[0].vector)
    return None


def get_embeddings_for_objects(object_ids: list[int]) -> dict[int, np.ndarray]:
    """Get embeddings for multiple object_ids from Qdrant."""
    client = get_qdrant_client()
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=object_ids,
        with_vectors=True,
    )
    return {p.id: np.array(p.vector) for p in points if p.vector}


def update_embedding(object_id: int, embedding: np.ndarray) -> bool:
    """Update the embedding for an object_id in Qdrant."""
    client = get_qdrant_client()
    try:
        # Get existing payload if any
        existing = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[object_id],
            with_payload=True,
            with_vectors=False,
        )
        payload = existing[0].payload if existing else {}

        # Upsert the point with new embedding
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=object_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            ],
        )
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update embedding for {object_id}: {e}")
        return False


def delete_objects_from_db(object_ids: list[int]) -> int:
    """Delete objects and their related events from the database by their IDs."""
    if not object_ids:
        return 0

    deleted = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for oid in object_ids:
                # First delete related events (foreign key constraint)
                cur.execute("DELETE FROM events WHERE object_id = %s", (oid,))
                # Then delete the object
                cur.execute("DELETE FROM objects WHERE object_id = %s", (oid,))
                deleted += cur.rowcount
        conn.commit()
    return deleted


def delete_objects_from_qdrant(object_ids: list[int]) -> int:
    """Delete objects from Qdrant by their IDs."""
    if not object_ids:
        return 0

    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=object_ids,
        )
        return len(object_ids)
    except Exception as e:
        print(f"[ERROR] Failed to delete from Qdrant: {e}")
        return 0


def combine_embeddings(
    registered_object_id: int,
    selected_object_ids: list[int],
    weight_existing: float = 0.5,
    cleanup: bool = True
) -> dict:
    """
    Combine selected embeddings with the registered entry's existing embedding.

    Workflow:
    1. Calculate centroid (average) of selected embeddings
    2. Fetch existing embedding for registered entry from Qdrant
    3. Calculate weighted average: final = weight * existing + (1-weight) * new_average
    4. Update Qdrant with final embedding
    5. Delete redundant entries from database and Qdrant (if cleanup=True)

    Args:
        registered_object_id: The main registered entry ID (to keep)
        selected_object_ids: List of object IDs selected by user to combine
        weight_existing: Weight for existing embedding (0.0-1.0), default 0.5
        cleanup: Whether to delete redundant entries after combining

    Returns:
        dict with success status and details
    """
    # Filter out the registered entry from selected (we handle it separately)
    other_ids = [oid for oid in selected_object_ids if oid != registered_object_id]

    if not other_ids:
        return {"success": False, "error": "No images selected to combine"}

    # Step 1: Get embeddings for selected images
    selected_embeddings = get_embeddings_for_objects(other_ids)

    if not selected_embeddings:
        return {"success": False, "error": "Could not retrieve selected embeddings"}

    # Calculate centroid (average) of selected embeddings
    vectors = list(selected_embeddings.values())
    new_average = np.mean(vectors, axis=0)

    # Normalize the new average
    norm = np.linalg.norm(new_average)
    if norm > 0:
        new_average = new_average / norm

    # Step 2: Fetch existing embedding for registered entry
    existing_embedding = get_embedding_for_object(registered_object_id)

    if existing_embedding is None:
        return {"success": False, "error": "Could not retrieve existing embedding for registered entry"}

    # Step 3: Calculate weighted average
    # final = weight_existing * existing + (1 - weight_existing) * new_average
    final_embedding = weight_existing * existing_embedding + (1 - weight_existing) * new_average

    # Normalize final embedding
    norm = np.linalg.norm(final_embedding)
    if norm > 0:
        final_embedding = final_embedding / norm

    # Step 4: Update Qdrant with final embedding
    success = update_embedding(registered_object_id, final_embedding)

    if not success:
        return {"success": False, "error": "Failed to update embedding in Qdrant"}

    # Step 5: Data cleanup - delete redundant entries
    deleted_db = 0
    deleted_qdrant = 0

    if cleanup and other_ids:
        deleted_db = delete_objects_from_db(other_ids)
        deleted_qdrant = delete_objects_from_qdrant(other_ids)

    # Clear cluster cache since embeddings changed
    global _cluster_cache
    _cluster_cache = {"labels": None, "point_ids": None, "n_clusters": None}

    return {
        "success": True,
        "object_id": registered_object_id,
        "combined_count": len(vectors),
        "weight_existing": weight_existing,
        "deleted_from_db": deleted_db,
        "deleted_from_qdrant": deleted_qdrant,
        "message": f"Combined {len(vectors)} images. Deleted {deleted_db} redundant entries."
    }


def get_non_ignored_cluster_members(main_object_id: int) -> list[int]:
    """
    Get all cluster members for main_object_id, excluding ignored ones.
    Returns list of object_ids that are NOT ignored.
    """
    # Get all cluster members
    cluster_members = get_cluster_for_point(main_object_id)

    if not cluster_members:
        return []

    # Filter out ignored ones by checking database
    non_ignored = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for oid in cluster_members:
                cur.execute("""
                    SELECT object_attributes->>'display_name'
                    FROM objects WHERE object_id = %s
                """, (oid,))
                row = cur.fetchone()
                # Include if no display_name or display_name is not 'ignored'
                if not row or row[0] != 'ignored':
                    non_ignored.append(oid)

    return non_ignored


def create_gold_embedding(main_object_id: int) -> dict:
    """
    Create Gold Embedding: average of ALL remaining (non-ignored) images in cluster,
    excluding the main image.

    Stores the original main embedding in Qdrant payload for Perfect to use later.

    Args:
        main_object_id: The registered entry ID (whose embedding will be updated)

    Returns:
        dict with success status and details
    """
    # Get all non-ignored cluster members, excluding main
    cluster_members = get_non_ignored_cluster_members(main_object_id)
    other_ids = [oid for oid in cluster_members if oid != main_object_id]

    if not other_ids:
        return {"success": False, "error": "No remaining images to combine (all ignored?)"}

    # Get original main embedding BEFORE we modify it
    original_main = get_embedding_for_object(main_object_id)
    if original_main is None:
        return {"success": False, "error": "Could not retrieve main object's embedding"}

    # Get embeddings for all other images
    embeddings = get_embeddings_for_objects(other_ids)

    if not embeddings:
        return {"success": False, "error": "Could not retrieve embeddings"}

    # Calculate Gold (average of all others)
    vectors = list(embeddings.values())
    gold_embedding = np.mean(vectors, axis=0)

    # Normalize
    norm = np.linalg.norm(gold_embedding)
    if norm > 0:
        gold_embedding = gold_embedding / norm

    # Store original embedding in Qdrant payload, then update with Gold
    client = get_qdrant_client()
    try:
        # Get existing payload
        existing = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[main_object_id],
            with_payload=True,
            with_vectors=False,
        )
        payload = existing[0].payload if existing else {}

        # Store original embedding for Perfect to use later
        payload["original_embedding"] = original_main.tolist()
        payload["gold_applied"] = True

        # Update with Gold embedding
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=main_object_id,
                    vector=gold_embedding.tolist(),
                    payload=payload,
                )
            ],
        )

        # Clear cluster cache
        global _cluster_cache
        _cluster_cache = {"labels": None, "point_ids": None, "n_clusters": None}

        return {
            "success": True,
            "object_id": main_object_id,
            "combined_count": len(vectors),
            "gold_applied": True,
            "message": f"Gold embedding created from {len(vectors)} images"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to update embedding: {e}"}


def create_perfect_embedding(main_object_id: int) -> dict:
    """
    Create Perfect Embedding: combines Gold (current) with Original main embedding.
    Then removes all other images in the cluster (merged into main).

    Formula: perfect = (gold + original_main) / 2

    Requires Gold to have been applied first (original_embedding stored in payload).

    Args:
        main_object_id: The registered entry ID

    Returns:
        dict with success status and details
    """
    client = get_qdrant_client()

    # Get all cluster members BEFORE we modify anything (for cleanup)
    cluster_members = get_non_ignored_cluster_members(main_object_id)
    other_ids = [oid for oid in cluster_members if oid != main_object_id]

    # Get current point with payload and vector
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[main_object_id],
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        return {"success": False, "error": "Could not find main object in Qdrant"}

    point = points[0]
    payload = point.payload or {}

    # Check if Gold was applied
    if not payload.get("gold_applied"):
        return {"success": False, "error": "Gold embedding must be applied first"}

    # Get original embedding from payload
    original_embedding_list = payload.get("original_embedding")
    if not original_embedding_list:
        return {"success": False, "error": "Original embedding not found. Apply Gold first."}

    original_embedding = np.array(original_embedding_list)
    gold_embedding = np.array(point.vector)

    # Calculate Perfect = (Gold + Original) / 2
    perfect_embedding = (gold_embedding + original_embedding) / 2

    # Normalize
    norm = np.linalg.norm(perfect_embedding)
    if norm > 0:
        perfect_embedding = perfect_embedding / norm

    # Update payload - remove original_embedding, mark perfect as applied
    payload.pop("original_embedding", None)
    payload["gold_applied"] = False
    payload["perfect_applied"] = True

    try:
        # Update main with perfect embedding
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=main_object_id,
                    vector=perfect_embedding.tolist(),
                    payload=payload,
                )
            ],
        )

        # Delete all other images from Qdrant and DB (they're now merged into main)
        deleted_qdrant = 0
        deleted_db = 0
        print(f"[Perfect] About to delete {len(other_ids)} other IDs: {other_ids[:5]}...")
        if other_ids:
            deleted_qdrant = delete_objects_from_qdrant(other_ids)
            print(f"[Perfect] Deleted {deleted_qdrant} from Qdrant")
            deleted_db = delete_objects_from_db(other_ids)
            print(f"[Perfect] Deleted {deleted_db} from DB")

        # Clear cluster cache
        global _cluster_cache
        _cluster_cache = {"labels": None, "point_ids": None, "n_clusters": None}

        return {
            "success": True,
            "object_id": main_object_id,
            "perfect_applied": True,
            "deleted_count": len(other_ids),
            "message": f"Perfect embedding created. Merged {len(other_ids)} images into main."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to update embedding: {e}"}


def combine_cluster_to_main(main_object_id: int) -> dict:
    """
    One-step combine: creates gold embedding then perfect embedding.
    Combines all cluster members into the main entry.

    Args:
        main_object_id: The registered entry ID

    Returns:
        dict with success status and details
    """
    # Step 1: Create Gold embedding
    gold_result = create_gold_embedding(main_object_id)
    if not gold_result.get("success"):
        return gold_result

    # Step 2: Create Perfect embedding (combines gold with original)
    perfect_result = create_perfect_embedding(main_object_id)
    if not perfect_result.get("success"):
        return {"success": False, "error": f"Gold succeeded but Perfect failed: {perfect_result.get('error')}"}

    return {
        "success": True,
        "object_id": main_object_id,
        "combined_count": gold_result.get("combined_count", 0),
        "deleted_count": perfect_result.get("deleted_count", 0),
        "message": f"Combined {gold_result.get('combined_count', 0)} faces into main entry"
    }


# ============== FastAPI Web Application ==============

app = FastAPI(title="Visitor Face Clustering Utility")

# Mount static files for photos using configured URL prefix
app.mount("/" + PHOTOS_PREFIX, StaticFiles(directory=str(PHOTOS_DIR)), name="photos")

# Templates
templates = Jinja2Templates(directory="templates")


def get_visitor_details(object_id: int) -> dict | None:
    """Get full visitor details from PostgreSQL."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT object_attributes FROM objects WHERE object_id = %s",
                (object_id,)
            )
            row = cur.fetchone()
            if row:
                return parse_attributes(row[0])
    return None


def is_ignored(object_id: int) -> bool:
    """Check if an object has display_name = 'ignored'."""
    details = get_visitor_details(object_id)
    if details and details.get("display_name") == "ignored":
        return True
    return False


def get_image_path_for_object(object_id: int) -> str | None:
    """Get image_path from PostgreSQL for a given object_id, with fallbacks."""
    details = get_visitor_details(object_id)

    if details and details.get("image_path"):
        img_path = details["image_path"]
        # Remove leading slash for file check
        check_path = img_path.lstrip("/")
        if os.path.exists(check_path):
            return img_path
        # Try resolving via config
        resolved = _resolve_photo_path(check_path)
        if resolved and os.path.exists(resolved):
            return "/" + PHOTOS_PREFIX + "/" + resolved[len(str(PHOTOS_DIR)) + 1:]

    # Fallback: search in unknown/recognized folders
    for folder in ["unknown", "recognized"]:
        folder_path = PHOTOS_DIR / folder / str(object_id)
        if folder_path.exists():
            # Find any image in the folder (recursively)
            for img in folder_path.rglob("*.jpg"):
                # Convert file path to URL path (PHOTOS_DIR -> PHOTOS_PREFIX)
                relative_path = img.relative_to(PHOTOS_DIR)
                return "/" + PHOTOS_PREFIX + "/" + str(relative_path)
            for img in folder_path.rglob("*.png"):
                relative_path = img.relative_to(PHOTOS_DIR)
                return "/" + PHOTOS_PREFIX + "/" + str(relative_path)

    return None


def _get_all_ignored_ids() -> set[int]:
    """Batch-load all ignored object IDs for dashboard performance."""
    ignored = set()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT object_id FROM objects
                WHERE object_attributes->>'display_name' = 'ignored'
            """)
            for row in cur.fetchall():
                ignored.add(row[0])
    return ignored


def _get_all_image_paths() -> dict[int, str]:
    """
    Batch-load all image paths for dashboard performance.
    Gets paths from DB first, then fills in unknown visitors by scanning filesystem.
    """
    paths = {}

    # 1. Get paths from DB (for registered visitors)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT object_id, object_attributes->>'image_path'
                FROM objects
                WHERE object_attributes->>'image_path' IS NOT NULL
            """)
            for row in cur.fetchall():
                if row[1]:
                    paths[row[0]] = row[1]

    # 2. Scan filesystem for unknown visitors (paths not in DB)
    # Structure: {PHOTOS_DIR}/unknown/{object_id}/{year}/{month}/{day}/{timestamp}.jpg
    unknown_dir = PHOTOS_DIR / "unknown"
    if unknown_dir.exists():
        for obj_folder in unknown_dir.iterdir():
            if obj_folder.is_dir():
                try:
                    obj_id = int(obj_folder.name)
                    if obj_id not in paths:
                        # Find any jpg in this folder
                        for img in obj_folder.rglob("*.jpg"):
                            # Store as DB-style path (using PHOTOS_PREFIX)
                            relative_path = img.relative_to(PHOTOS_DIR)
                            paths[obj_id] = PHOTOS_PREFIX + "/" + str(relative_path)
                            break  # Only need one image per object
                except ValueError:
                    pass  # Skip non-numeric folder names

    return paths


def _is_clustering_ready() -> bool:
    """Check if cluster cache is available (memory or disk)."""
    if _cluster_cache["labels"] is not None:
        return True
    # Try loading from disk without running clustering
    return _load_cache_from_disk()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, refresh: bool = False, sort: str = "similarity"):
    """Main dashboard page using K-means clustering."""
    start_time = time.time()

    # Refresh clusters if requested (in background)
    if refresh:
        start_background_clustering()

    # Check if clustering is ready - if not, start it and show loading state
    clustering_ready = _is_clustering_ready()

    if not clustering_ready:
        # Start background clustering if not already running
        if not _task_status.running:
            start_background_clustering()

        # Return loading page
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "visitors": [],
            "cluster_info": {"n_clusters": 0, "total_points": 0},
            "current_sort": sort,
            "loading": True,  # Flag for template to show loading state
        })

    visitors = get_visitors()
    cluster_info = get_cluster_info()

    # Batch-load data for performance (single DB queries instead of thousands)
    ignored_ids = _get_all_ignored_ids()
    all_image_paths = _get_all_image_paths()

    # Pre-load cluster assignments once (from cache, no blocking)
    all_clusters = cluster_embeddings()

    # Pre-load frontal cache
    _load_frontal_cache()

    print(f"[Dashboard] Loaded {len(ignored_ids)} ignored, {len(all_image_paths)} paths, {len(_frontal_cache)} frontal in {time.time() - start_time:.2f}s")

    # Build reverse cluster lookup: cluster_label -> list of point_ids
    cluster_to_points = {}
    for pid, label in all_clusters.items():
        if label not in cluster_to_points:
            cluster_to_points[label] = []
        cluster_to_points[label].append(pid)

    visitor_data = []
    for name, object_id in visitors.items():
        # Use batch-loaded image path instead of DB query
        details = {"image_path": all_image_paths.get(object_id)}

        # Get all points in the same cluster as this visitor (from pre-loaded data)
        my_cluster = all_clusters.get(object_id)
        similar_ids = cluster_to_points.get(my_cluster, []) if my_cluster is not None else []

        # Count total valid cluster members (excluding self, ignored)
        total_valid = sum(1 for pid in similar_ids if pid != object_id and pid not in ignored_ids and pid in all_image_paths)

        # Get main image
        main_image = None
        if details and details.get("image_path"):
            img_path = details["image_path"]
            # Convert DB path prefix to URL prefix
            if img_path.startswith(PHOTOS_PREFIX + "/"):
                main_image = "/" + PHOTOS_PREFIX + "/" + img_path[len(PHOTOS_PREFIX) + 1:]
            elif img_path.startswith("/" + PHOTOS_PREFIX + "/"):
                main_image = "/" + PHOTOS_PREFIX + "/" + img_path[len(PHOTOS_PREFIX) + 2:]
            else:
                main_image = "/" + PHOTOS_PREFIX + "/" + img_path

        # Get preview images (top 4 for card display - no frontal check for speed)
        preview_images = []
        for pid in similar_ids:
            if len(preview_images) >= 4:
                break
            if pid == object_id:
                continue
            if pid in ignored_ids:
                continue

            img_path = all_image_paths.get(pid)
            if not img_path:
                continue

            # Normalize path to URL format
            if img_path.startswith(PHOTOS_PREFIX):
                img_path = "/" + PHOTOS_PREFIX + "/" + img_path[len(PHOTOS_PREFIX) + 1:]
            elif not img_path.startswith("/"):
                img_path = "/" + PHOTOS_PREFIX + "/" + img_path
            preview_images.append({"id": pid, "path": img_path})

        visitor_data.append({
            "name": name,
            "object_id": object_id,
            "main_image": main_image,
            "similar_count": total_valid,  # Total cluster size
            "similar_images": preview_images,  # Just preview (4 for cards)
            "details": details,
        })

    # Sort visitor data
    if sort == "alphabetic":
        visitor_data.sort(key=lambda x: x["name"].lower())
    else:  # similarity (default) - sort by similar_count descending
        visitor_data.sort(key=lambda x: x["similar_count"], reverse=True)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "visitors": visitor_data,
        "cluster_info": cluster_info,
        "current_sort": sort,
        "loading": False,
    })


@app.get("/api/visitors")
async def api_visitors():
    """API endpoint for visitor list."""
    return get_visitors()


@app.get("/api/visitor/{object_id}")
async def api_visitor(object_id: int):
    """API endpoint for single visitor with cluster members."""
    details = get_visitor_details(object_id)
    similar_ids = get_cluster_for_point(object_id)
    return {
        "object_id": object_id,
        "details": details,
        "cluster_members": similar_ids,
    }


@app.get("/api/cluster-images/{object_id}")
async def api_cluster_images(object_id: int, frontal_only: bool = True):
    """
    Get ALL images in the same cluster as object_id.
    Applies frontal face filter if enabled and frontal_only=True.
    Returns images with cached frontal check, running checks for uncached ones.
    """
    start_time = time.time()

    # Get cluster for this object
    similar_ids = get_cluster_for_point(object_id)

    # Batch-load data
    ignored_ids = _get_all_ignored_ids()
    all_image_paths = _get_all_image_paths()
    _load_frontal_cache()

    images = []
    frontal_checked = 0
    frontal_passed = 0
    frontal_check_enabled = FACE_FILTER_CONFIG.get("enabled", False) and frontal_only

    for pid in similar_ids:
        if pid == object_id:
            continue
        if pid in ignored_ids:
            continue

        img_path = all_image_paths.get(pid)
        if not img_path:
            continue

        # Apply frontal check if enabled
        if frontal_check_enabled:
            frontal_checked += 1
            if not is_frontal_cached(pid, img_path):
                continue
            frontal_passed += 1

        # Normalize path to URL format
        if img_path.startswith(PHOTOS_PREFIX):
            img_path = "/" + PHOTOS_PREFIX + "/" + img_path[len(PHOTOS_PREFIX) + 1:]
        elif not img_path.startswith("/"):
            img_path = "/" + PHOTOS_PREFIX + "/" + img_path
        images.append({"id": pid, "path": img_path})

    elapsed = time.time() - start_time
    print(f"[ClusterImages] object={object_id}, total={len(similar_ids)}, checked={frontal_checked}, passed={frontal_passed}, returned={len(images)} in {elapsed:.2f}s")

    return {
        "object_id": object_id,
        "total_in_cluster": len(similar_ids),
        "images": images,
        "frontal_filter_enabled": frontal_check_enabled,
    }


@app.get("/api/cluster-info")
async def api_cluster_info():
    """Get clustering statistics."""
    return get_cluster_info()


@app.get("/api/task-status")
async def api_task_status():
    """Get background task status for progress bar."""
    return get_task_status()


@app.post("/api/refresh-clusters")
async def api_refresh_clusters():
    """Start re-clustering in background."""
    started = start_background_clustering()
    if started:
        return {"success": True, "message": "Clustering started in background"}
    else:
        return {"success": False, "message": "Clustering already in progress"}


@app.post("/api/ignore")
async def api_ignore(request: Request):
    """Mark selected object_ids as ignored by setting display_name to 'ignored'."""
    try:
        data = await request.json()
        object_ids = data.get("object_ids", [])

        if not object_ids:
            return {"success": False, "error": "No object IDs provided"}

        ignored_count = 0
        skipped_ids = []

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for oid in object_ids:
                    # Check if this object has promoted_to_known = true
                    cur.execute("""
                        SELECT object_attributes->>'promoted_to_known'
                        FROM objects WHERE object_id = %s
                    """, (oid,))
                    row = cur.fetchone()

                    if row and row[0] == 'true':
                        skipped_ids.append(oid)
                        continue

                    cur.execute("""
                        UPDATE objects
                        SET object_attributes = object_attributes || '{"display_name": "ignored"}'::jsonb
                        WHERE object_id = %s
                    """, (oid,))
                    ignored_count += 1
                conn.commit()

        if skipped_ids:
            return {
                "success": True,
                "updated": ignored_count,
                "skipped": skipped_ids,
                "message": f"Ignored {ignored_count} items. Skipped {len(skipped_ids)} (protected)."
            }

        return {"success": True, "updated": ignored_count}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/combine")
async def api_combine(request: Request):
    """
    Combine all cluster faces into the main registered entry.
    This does Gold + Perfect in one step.
    """
    try:
        data = await request.json()
        registered_object_id = data.get("registered_object_id")

        if not registered_object_id:
            return {"success": False, "error": "registered_object_id is required"}

        result = combine_cluster_to_main(registered_object_id)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
