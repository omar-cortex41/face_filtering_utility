from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os

from main import (
    get_db_connection,
    get_visitors,
    get_cluster_for_point,
    get_cluster_info,
    cluster_embeddings,
    parse_attributes,
    is_frontal_face,
    combine_embeddings,
    create_gold_embedding,
    create_perfect_embedding,
    PHOTOS_DIR,
)

app = FastAPI(title="Visitor Embeddings Dashboard")

# Mount static files for photos
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

# Templates
templates = Jinja2Templates(directory="templates")

PHOTOS_DIR = Path(PHOTOS_DIR)


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

    # Fallback: search in unknown/recognized folders
    for folder in ["unknown", "recognized"]:
        folder_path = PHOTOS_DIR / folder / str(object_id)
        if folder_path.exists():
            # Find any image in the folder (recursively)
            for img in folder_path.rglob("*.jpg"):
                # Convert file path to URL path (PHOTOS_DIR -> /photos)
                relative_path = img.relative_to(PHOTOS_DIR)
                return "/photos/" + str(relative_path)
            for img in folder_path.rglob("*.png"):
                relative_path = img.relative_to(PHOTOS_DIR)
                return "/photos/" + str(relative_path)

    return None


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, refresh: bool = False):
    """Main dashboard page using K-means clustering."""
    # Refresh clusters if requested
    if refresh:
        cluster_embeddings(force_refresh=True)

    visitors = get_visitors()
    cluster_info = get_cluster_info()

    visitor_data = []
    for name, object_id in visitors.items():
        details = get_visitor_details(object_id)

        # Get all points in the same cluster as this visitor
        similar_ids = get_cluster_for_point(object_id)

        # Get main image
        main_image = None
        if details and details.get("image_path"):
            img_path = details["image_path"]
            if img_path.startswith("photos/"):
                main_image = "/" + img_path
            elif img_path.startswith("/photos/"):
                main_image = img_path

        # Get similar images - each point_id = object_id in PostgreSQL
        # Filter out: main image, ignored images, non-frontal faces, and missing images
        similar_images = []
        for pid in similar_ids:
            # Skip the main image itself (it's shown separately)
            if pid == object_id:
                continue

            # Skip ignored objects
            if is_ignored(pid):
                continue

            img_path = get_image_path_for_object(pid)

            # Skip entries without images
            if not img_path:
                continue

            # Check if frontal face (skip non-frontal)
            if not is_frontal_face(img_path):
                continue

            # Normalize path - ensure it starts with /photos
            if img_path.startswith("/photos"):
                img_path = img_path  # already correct
            elif img_path.startswith("photos"):
                img_path = "/" + img_path
            similar_images.append({"id": pid, "path": img_path})

        visitor_data.append({
            "name": name,
            "object_id": object_id,
            "main_image": main_image,
            "similar_count": len(similar_images),  # Count after filtering
            "similar_images": similar_images,
            "details": details,
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "visitors": visitor_data,
        "cluster_info": cluster_info,
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


@app.get("/api/cluster-info")
async def api_cluster_info():
    """Get clustering statistics."""
    return get_cluster_info()


@app.post("/api/refresh-clusters")
async def api_refresh_clusters():
    """Force re-clustering of all embeddings."""
    cluster_embeddings(force_refresh=True)
    return get_cluster_info()


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
                        # Skip - this is a known user's main image
                        skipped_ids.append(oid)
                        continue

                    # Update the display_name in object_attributes
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
                "message": f"Ignored {ignored_count} items. Skipped {len(skipped_ids)} (promoted to known user)."
            }

        return {"success": True, "updated": ignored_count}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/combine")
async def api_combine_embeddings(request: Request):
    """
    Combine selected embeddings with the registered entry.

    Workflow:
    1. Calculate centroid of SELECTED images
    2. Fetch existing embedding for registered entry from Qdrant
    3. Calculate weighted average of existing + new centroid
    4. Update Qdrant with final embedding
    5. Delete redundant entries from database and Qdrant

    Request body:
    - registered_object_id: The main registered entry ID (to keep)
    - selected_object_ids: List of object IDs selected by user to combine
    - weight_existing: Weight for existing embedding (0.0-1.0), default 0.5
    - cleanup: Whether to delete redundant entries (default true)
    """
    try:
        data = await request.json()
        registered_object_id = data.get("registered_object_id")
        selected_object_ids = data.get("selected_object_ids", [])
        weight_existing = data.get("weight_existing", 0.5)
        cleanup = data.get("cleanup", True)

        if not registered_object_id:
            return {"success": False, "error": "registered_object_id is required"}

        if not selected_object_ids:
            return {"success": False, "error": "selected_object_ids is required"}

        result = combine_embeddings(
            registered_object_id=registered_object_id,
            selected_object_ids=selected_object_ids,
            weight_existing=weight_existing,
            cleanup=cleanup
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/gold-embedding")
async def api_gold_embedding(request: Request):
    """
    Create Gold Embedding: average of ALL remaining (non-ignored) images, excluding main.
    No selection needed - operates on all non-ignored cluster members.
    Stores original main embedding for Perfect to use later.
    """
    try:
        data = await request.json()
        registered_object_id = data.get("registered_object_id")

        if not registered_object_id:
            return {"success": False, "error": "registered_object_id is required"}

        result = create_gold_embedding(registered_object_id)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/perfect-embedding")
async def api_perfect_embedding(request: Request):
    """
    Create Perfect Embedding: Gold + Original main.
    Requires Gold to have been applied first.
    """
    try:
        data = await request.json()
        registered_object_id = data.get("registered_object_id")

        if not registered_object_id:
            return {"success": False, "error": "registered_object_id is required"}

        result = create_perfect_embedding(registered_object_id)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

