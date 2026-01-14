import psycopg2
import json
from pathlib import Path
import yaml
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from qdrant_client import QdrantClient
import cv2

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
PHOTOS_DIR = config["photos"]["directory"]
FACE_FILTER_CONFIG = config.get("face_filter", {})


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
    Get all visitors with their object_id.
    Returns {name: object_id}.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT object_attributes->>'name' as name, object_id
                FROM objects
                WHERE object_attributes ? 'visitor_id'
            """)
            return {row[0]: row[1] for row in cur.fetchall() if row[0]}


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


def get_all_embeddings() -> tuple[list[int], np.ndarray]:
    """Retrieve all embeddings from Qdrant."""
    client = get_qdrant_client()

    # Scroll through all points
    all_points = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_vectors=True,
        )
        all_points.extend(points)
        if offset is None:
            break

    point_ids = [p.id for p in all_points]
    vectors = np.array([p.vector for p in all_points])

    return point_ids, vectors


def find_optimal_k(vectors: np.ndarray, min_k: int = 2, max_k: int = 50) -> int:
    """
    Find optimal number of clusters using silhouette score.
    """
    n_samples = len(vectors)
    max_k = min(max_k, n_samples - 1)  # Can't have more clusters than samples

    if max_k < min_k:
        return min_k

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        # Silhouette score requires at least 2 clusters with >1 sample
        if len(set(labels)) > 1:
            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_score = score
                best_k = k

    return best_k


def cluster_embeddings(force_refresh: bool = False) -> dict[int, int]:
    """
    Cluster all embeddings using K-means with auto-detected K.
    Returns: {point_id: cluster_label}
    """
    global _cluster_cache

    if not force_refresh and _cluster_cache["labels"] is not None:
        return dict(zip(_cluster_cache["point_ids"], _cluster_cache["labels"]))

    point_ids, vectors = get_all_embeddings()

    if len(vectors) < 2:
        return {pid: 0 for pid in point_ids}

    # Get number of known visitors as minimum K
    visitors = get_visitors()
    min_k = max(2, len(visitors))
    max_k = min(len(vectors) // 2, 100)  # Don't exceed half the samples

    # Find optimal K
    optimal_k = find_optimal_k(vectors, min_k=min_k, max_k=max_k)

    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    # Cache results
    _cluster_cache["labels"] = labels
    _cluster_cache["point_ids"] = point_ids
    _cluster_cache["n_clusters"] = optimal_k

    return dict(zip(point_ids, labels))


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


if __name__ == "__main__":
    analyze_user_embeddings(similarity_threshold=0.2)
