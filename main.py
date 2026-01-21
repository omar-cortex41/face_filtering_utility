"""
Visitor Face Clustering Utility - Main Application
Combines database operations, clustering, and web API in one file.
"""
import psycopg2
import json
from pathlib import Path
import yaml
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import cv2
import time
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
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

# MediaPipe Face Mesh for frontal face detection (468 landmarks)
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp_face_mesh = None
    MEDIAPIPE_AVAILABLE = False

# InsightFace for face verification (ArcFace embeddings)
try:
    from insightface.app import FaceAnalysis
    _insightface_app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    _insightface_app.prepare(ctx_id=0, det_size=(320, 320))
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    _insightface_app = None
    INSIGHTFACE_AVAILABLE = False

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
UI_CONFIG = config.get("ui", {})
PAGE_SIZE = int(UI_CONFIG.get("page_size", 20))


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


# ========= Frontal Face Filter (MediaPipe Face Mesh) =========

# 3D model points for head pose estimation (generic face model)
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# Face Mesh landmark indices for the 6 points above
_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def is_frontal_face(image_path: str) -> bool:
    """
    Check if face is frontal using MediaPipe Face Mesh (468 landmarks).
    Uses solvePnP to calculate head pose (yaw, pitch, roll).
    Returns True if frontal, False otherwise.
    """
    if not FACE_FILTER_CONFIG.get("enabled", True):
        return True

    if not MEDIAPIPE_AVAILABLE:
        print("[WARN] MediaPipe not available, skipping frontal check")
        return True

    mp_config = FACE_FILTER_CONFIG.get("mediapipe", {})
    yaw_max = float(mp_config.get("yaw_max", 15.0))    # degrees
    pitch_max = float(mp_config.get("pitch_max", 15.0))  # degrees
    min_confidence = float(mp_config.get("min_detection_confidence", 0.5))
    blur_min = float(FACE_FILTER_CONFIG.get("blur_min", 5.0))

    try:
        img_path = image_path.lstrip("/")
        if not Path(img_path).exists():
            return False

        frame = cv2.imread(img_path)
        if frame is None:
            return False

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_val < blur_min:
            return False

        # Detect face mesh with MediaPipe
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
        ) as face_mesh:
            results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0].landmark

        # Extract the 6 key points for pose estimation
        image_points = np.array([
            (landmarks[idx].x * w, landmarks[idx].y * h)
            for idx in _LANDMARK_INDICES
        ], dtype=np.float64)

        # Camera matrix (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # No lens distortion

        # Solve for head pose
        success, rotation_vec, translation_vec = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return False

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Get Euler angles (in degrees)
        # Using rotation matrix to extract yaw, pitch, roll
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

        if sy > 1e-6:
            pitch = np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))
            yaw = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
        else:
            pitch = np.degrees(np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1]))
            yaw = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))

        # Normalize pitch: frontal faces have pitch around -165° to -175°
        # Convert to deviation from frontal (0 = looking straight at camera)
        pitch_normalized = abs(abs(pitch) - 180) if abs(pitch) > 90 else abs(pitch)

        # Check if face is frontal (within thresholds)
        is_frontal = abs(yaw) <= yaw_max and pitch_normalized <= pitch_max

        return is_frontal

    except Exception as e:
        print(f"[WARN] Face Mesh frontal check failed for {image_path}: {e}")
        return False


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


def check_frontal(image_path: str) -> bool:
    """Check if face is frontal. No caching."""
    # If filter disabled, all faces are "frontal"
    if not FACE_FILTER_CONFIG.get("enabled", True):
        return True

    resolved_path = _resolve_photo_path(image_path)
    return is_frontal_face(resolved_path)


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


# ============== Qdrant Cosine Similarity Search ==============

# Similarity threshold from config (or default)
SIMILARITY_CONFIG = config.get("similarity", {})
SIMILARITY_THRESHOLD = float(SIMILARITY_CONFIG.get("threshold", 0.5))


def get_similar_faces(object_id: int, threshold: float = None, limit: int = 500) -> list[int]:
    """
    Get all similar face IDs for a given object using Qdrant cosine similarity.
    This is the core similarity function - uses the vector DB as intended.

    Args:
        object_id: The object_id to find similar faces for
        threshold: Cosine similarity threshold (0-1, higher = more similar)
        limit: Maximum number of results

    Returns:
        List of similar object_ids (including self)
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    client = get_qdrant_client()

    # Get the embedding for this object
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[object_id],
        with_vectors=True,
    )

    if not points or not points[0].vector:
        return [object_id]  # Return self if no embedding

    vector = points[0].vector

    # Query Qdrant for similar embeddings
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=limit,
        score_threshold=threshold,
    )

    # Return all matching point IDs
    similar_ids = [p.id for p in result.points]

    return similar_ids if similar_ids else [object_id]


def get_similar_faces_with_scores(object_id: int, threshold: float = None, limit: int = 500) -> dict[int, float]:
    """
    Get all similar face IDs with their similarity scores using Qdrant cosine similarity.

    Args:
        object_id: The object_id to find similar faces for
        threshold: Cosine similarity threshold (0-1, higher = more similar)
        limit: Maximum number of results

    Returns:
        Dict mapping object_id -> similarity score (0-1, higher = more similar)
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    client = get_qdrant_client()

    # Get the embedding for this object
    points = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[object_id],
        with_vectors=True,
    )

    if not points or not points[0].vector:
        return {object_id: 1.0}  # Return self with 100% similarity

    vector = points[0].vector

    # Query Qdrant for similar embeddings
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=limit,
        score_threshold=threshold,
    )

    # Return dict of id -> score
    scores = {p.id: p.score for p in result.points}

    return scores if scores else {object_id: 1.0}


def get_cluster_for_point(object_id: int) -> list[int]:
    """
    Get all point IDs similar to the given object_id using Qdrant cosine similarity.
    This replaces clustering - uses direct vector similarity search.
    """
    return get_similar_faces(object_id)


def get_cluster_info() -> dict:
    """Get info about the similarity search method."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(QDRANT_COLLECTION)
        total = info.points_count
    except Exception:
        total = 0

    return {
        "method": "Qdrant Cosine Similarity",
        "threshold": SIMILARITY_THRESHOLD,
        "total_points": total,
    }


def start_background_clustering():
    """No-op for backwards compatibility. Qdrant similarity is instant."""
    return True


def get_task_status() -> dict:
    """Return ready status - Qdrant similarity is instant, no background task needed."""
    return {
        "running": False,
        "task_name": "",
        "progress": 100,
        "message": "Ready (Qdrant cosine similarity)",
        "error": None,
    }


def cluster_embeddings(force_refresh: bool = False, background: bool = False) -> dict[int, int]:
    """
    Backwards compatibility function.
    With Qdrant similarity, we don't pre-cluster - similarity is computed on-demand.
    Returns empty dict since we use get_similar_faces() directly now.
    """
    return {}





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


def delete_non_frontal_faces(object_ids: list[int], all_image_paths: dict) -> dict:
    """
    Check faces and delete those that fail the frontal face filter.
    Deletes from both PostgreSQL (events + objects) and Qdrant.

    Returns dict with:
    - deleted_count: number of faces deleted
    - deleted_ids: list of deleted object IDs
    """
    if not object_ids:
        return {"deleted_count": 0, "deleted_ids": []}

    ids_to_delete = []

    for oid in object_ids:
        img_path = all_image_paths.get(oid)
        if not img_path:
            continue

        # Check if face is frontal
        if not check_frontal(img_path):
            ids_to_delete.append(oid)

    if not ids_to_delete:
        return {"deleted_count": 0, "deleted_ids": []}

    print(f"[AutoCleanup] Deleting {len(ids_to_delete)} non-frontal faces: {ids_to_delete}")

    # Delete from PostgreSQL (events first due to foreign key constraint)
    deleted_db = 0
    deleted_events = 0
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for oid in ids_to_delete:
                    # Delete related events first
                    cur.execute("DELETE FROM events WHERE object_id = %s", (oid,))
                    deleted_events += cur.rowcount
                    # Then delete the object
                    cur.execute("DELETE FROM objects WHERE object_id = %s", (oid,))
                    deleted_db += cur.rowcount
                conn.commit()
        print(f"[AutoCleanup] Deleted {deleted_db} from objects, {deleted_events} from events")
    except Exception as e:
        print(f"[AutoCleanup] Error deleting from PostgreSQL: {e}")

    # Delete from Qdrant
    deleted_qdrant = delete_objects_from_qdrant(ids_to_delete)
    print(f"[AutoCleanup] Deleted {deleted_qdrant} from Qdrant")

    return {"deleted_count": len(ids_to_delete), "deleted_ids": ids_to_delete}


def combine_selected_faces(main_object_id: int, selected_ids: list[int]) -> dict:
    """
    Combine only the user-selected faces into the main entry's embedding.

    Args:
        main_object_id: The registered entry ID (whose embedding will be updated)
        selected_ids: List of object_ids selected by the user in the UI

    Returns:
        dict with success status and details
    """
    if not selected_ids:
        return {"success": False, "error": "No faces selected to combine"}

    # Get original main embedding
    original_main = get_embedding_for_object(main_object_id)
    if original_main is None:
        return {"success": False, "error": "Could not retrieve main object's embedding"}

    # Get embeddings for selected faces
    embeddings = get_embeddings_for_objects(selected_ids)
    if not embeddings:
        return {"success": False, "error": "Could not retrieve embeddings for selected faces"}

    # Calculate average of selected embeddings (Gold)
    vectors = list(embeddings.values())
    gold_embedding = np.mean(vectors, axis=0)

    # Normalize
    norm = np.linalg.norm(gold_embedding)
    if norm > 0:
        gold_embedding = gold_embedding / norm

    # Calculate Perfect embedding: average of gold + original
    perfect_embedding = np.mean([gold_embedding, original_main], axis=0)
    norm = np.linalg.norm(perfect_embedding)
    if norm > 0:
        perfect_embedding = perfect_embedding / norm

    # Update main object's embedding in Qdrant
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

        # Store original embedding and mark as combined
        payload["original_embedding"] = original_main.tolist()
        payload["combined"] = True
        payload["combined_count"] = len(selected_ids)

        # Update with Perfect embedding
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

        # Delete combined images from PostgreSQL (events first due to FK constraint)
        deleted_db = 0
        deleted_events = 0
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for oid in selected_ids:
                    cur.execute("DELETE FROM events WHERE object_id = %s", (oid,))
                    deleted_events += cur.rowcount
                    cur.execute("DELETE FROM objects WHERE object_id = %s", (oid,))
                    deleted_db += cur.rowcount
                conn.commit()

        # Delete combined images from Qdrant
        deleted_qdrant = delete_objects_from_qdrant(selected_ids)

        print(f"[Combine] Deleted {deleted_db} objects, {deleted_events} events from DB, {deleted_qdrant} from Qdrant")

        return {
            "success": True,
            "object_id": main_object_id,
            "combined_count": len(selected_ids),
            "deleted_from_db": deleted_db,
            "deleted_from_qdrant": deleted_qdrant,
            "message": f"Combined {len(selected_ids)} face(s) and deleted from database"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to update embedding: {str(e)}"}


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


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, sort: str = "similarity"):
    """Main dashboard page - loads immediately, faces load async via API."""
    cluster_info = get_cluster_info()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "cluster_info": cluster_info,
        "current_sort": sort,
    })

@app.get("/api/visitors-data-stream")
async def api_visitors_data_stream(sort: str = "similarity", frontal_only: bool = True):
    """SSE endpoint that streams progress as visitors are processed.
    Sends progress events, then final data event."""

    def generate():
        visitors = get_visitors()
        total_count = len(visitors)

        # Send initial count
        yield f"data: {json.dumps({'type': 'start', 'total': total_count})}\n\n"

        # Batch-load data for performance
        ignored_ids = _get_all_ignored_ids()
        all_image_paths = _get_all_image_paths()

        visitor_data = []
        all_non_frontal_ids = []
        processed = 0

        for name, object_id in visitors.items():
            details = {"image_path": all_image_paths.get(object_id)}
            similar_ids = get_similar_faces(object_id)

            total_valid = sum(1 for pid in similar_ids if pid != object_id and pid not in ignored_ids and pid in all_image_paths)

            main_image = None
            main_image_qualified = True
            if details and details.get("image_path"):
                img_path = details["image_path"]
                main_image_qualified = bool(check_frontal(img_path))
                if img_path.startswith(PHOTOS_PREFIX + "/"):
                    main_image = "/" + PHOTOS_PREFIX + "/" + img_path[len(PHOTOS_PREFIX) + 1:]
                elif img_path.startswith("/" + PHOTOS_PREFIX + "/"):
                    main_image = "/" + PHOTOS_PREFIX + "/" + img_path[len(PHOTOS_PREFIX) + 2:]
                else:
                    main_image = "/" + PHOTOS_PREFIX + "/" + img_path

            preview_images = []
            for pid in similar_ids:
                if len(preview_images) >= 4:
                    break
                if pid == object_id or pid in ignored_ids:
                    continue
                img_path = all_image_paths.get(pid)
                if not img_path:
                    continue
                if frontal_only:
                    if not check_frontal(img_path):
                        all_non_frontal_ids.append(pid)
                        continue
                if img_path.startswith(PHOTOS_PREFIX):
                    img_path = "/" + PHOTOS_PREFIX + "/" + img_path[len(PHOTOS_PREFIX) + 1:]
                elif not img_path.startswith("/"):
                    img_path = "/" + PHOTOS_PREFIX + "/" + img_path
                preview_images.append({"id": pid, "path": img_path})

            visitor_data.append({
                "name": name,
                "object_id": object_id,
                "main_image": main_image,
                "main_image_qualified": main_image_qualified,
                "similar_count": total_valid,
                "similar_images": preview_images,
            })

            processed += 1
            percent = int((processed / total_count) * 100)
            # Send progress update every visitor
            yield f"data: {json.dumps({'type': 'progress', 'processed': processed, 'total': total_count, 'percent': percent})}\n\n"

        # Auto-delete non-frontal faces
        deleted_count = 0
        if frontal_only and all_non_frontal_ids:
            unique_non_frontal = list(set(all_non_frontal_ids))
            cleanup_result = delete_non_frontal_faces(unique_non_frontal, all_image_paths)
            deleted_count = cleanup_result.get("deleted_count", 0)

        # Sort
        if sort == "alphabetic":
            visitor_data.sort(key=lambda x: x["name"].lower())
        else:
            visitor_data.sort(key=lambda x: x["similar_count"], reverse=True)

        # Get total faces count from Qdrant
        try:
            client = get_qdrant_client()
            total_faces = client.get_collection(QDRANT_COLLECTION).points_count
        except Exception:
            total_faces = 0

        # Send final data
        yield f"data: {json.dumps({'type': 'complete', 'visitors': visitor_data, 'total': len(visitor_data), 'total_faces': total_faces, 'page_size': PAGE_SIZE, 'auto_deleted_count': deleted_count})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
    Get ALL images in the same cluster as object_id with similarity scores.
    When frontal_only=True, non-frontal faces are auto-deleted from PostgreSQL and Qdrant.
    """
    start_time = time.time()

    # Get cluster for this object WITH similarity scores
    similar_scores = get_similar_faces_with_scores(object_id)

    # Batch-load data
    ignored_ids = _get_all_ignored_ids()
    all_image_paths = _get_all_image_paths()

    images = []
    non_frontal_ids = []
    deleted_count = 0

    for pid, score in similar_scores.items():
        if pid == object_id:
            continue
        if pid in ignored_ids:
            continue

        img_path = all_image_paths.get(pid)
        if not img_path:
            continue

        # Check frontal - when enabled, non-frontal faces are auto-deleted
        if frontal_only:
            if not check_frontal(img_path):
                non_frontal_ids.append(pid)
                continue

        # Normalize path to URL format
        if img_path.startswith(PHOTOS_PREFIX):
            img_path = "/" + PHOTOS_PREFIX + "/" + img_path[len(PHOTOS_PREFIX) + 1:]
        elif not img_path.startswith("/"):
            img_path = "/" + PHOTOS_PREFIX + "/" + img_path

        # Convert score (0-1) to percentage and include in response
        similarity_percent = round(score * 100, 1)
        images.append({"id": pid, "path": img_path, "similarity": similarity_percent})

    # Auto-delete non-frontal faces when filter is enabled
    if frontal_only and non_frontal_ids:
        cleanup_result = delete_non_frontal_faces(non_frontal_ids, all_image_paths)
        deleted_count = cleanup_result.get("deleted_count", 0)

    elapsed = time.time() - start_time
    print(f"[ClusterImages] object={object_id}, total={len(similar_scores)}, returned={len(images)}, auto_deleted={deleted_count} in {elapsed:.2f}s")

    return {
        "object_id": object_id,
        "total_in_cluster": len(similar_scores),
        "images": images,
        "frontal_filter_enabled": frontal_only,
        "auto_deleted_count": deleted_count,
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


@app.post("/api/combine")
async def api_combine(request: Request):
    """
    Combine selected faces into the main registered visitor entry.
    - combine_ids: faces to combine with the main visitor
    """
    try:
        data = await request.json()
        registered_object_id = data.get("registered_object_id")
        combine_ids = data.get("combine_ids", [])

        if not registered_object_id:
            return {"success": False, "error": "registered_object_id is required"}

        if not combine_ids:
            return {"success": False, "error": "No faces to combine"}

        # Combine the selected faces
        result = combine_selected_faces(registered_object_id, combine_ids)

        if result.get("success"):
            result["message"] = f"Combined {result.get('combined_count', 0)} face(s)"

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
