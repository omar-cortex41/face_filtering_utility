
## 1 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Technology Stack](#2-architecture--technology-stack)
3. [Database Schema](#3-database-schema)
4. [Setup Instructions](#4-setup-instructions)
5. [Core Components](#5-core-components)
6. [API Documentation](#6-api-documentation)
7. [User Interface](#7-user-interface)
8. [Deployment Process](#8-deployment-process)
9. [Testing & Validation](#9-testing--validation)

---


### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser                               │
│                    (dashboard.html)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/HTTPS
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│                        (app.py)                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────┐ │
│  │ HTML Routes  │  │ API Endpoints │  │ Static File Server    │ │
│  │ GET /        │  │ GET /api/*    │  │ /photos/*             │ │
│  └──────────────┘  └───────────────┘  └───────────────────────┘ │
└──────────────┬────────────────────────────────┬─────────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────────┐    ┌───────────────────────────────┐
│     PostgreSQL DB        │    │        Qdrant Vector DB        │
│   (analyticschronicles)  │    │  (visitor_embeddings_office)   │
│                          │    │                                 │
│  ┌────────────────────┐  │    │  ┌─────────────────────────┐   │
│  │   objects table    │  │    │  │   Face Embeddings       │   │
│  │ - object_id (PK)   │  │    │  │ - point_id = object_id  │   │
│  │ - object_attributes│  │    │  │ - vector (512/768-dim)  │   │
│  └────────────────────┘  │    │  └─────────────────────────┘   │
└──────────────────────────┘    └───────────────────────────────┘
```



### 2.3 Data Flow

1. **User Request**: Browser requests dashboard at `/` or API endpoints
2. **Visitor Retrieval**: FastAPI queries PostgreSQL for visitors with `visitor_id` attribute
3. **Embedding Search**: For each visitor, retrieve embedding from Qdrant and find similar vectors
4. **Image Resolution**: Map object IDs to image paths from database or file system fallback
5. **Template Rendering**: Jinja2 renders HTML with visitor data and similar images
6. **Response**: HTML page or JSON response returned to browser

---

## 3. Database Schema

### 3.1 Objects Table Structure

The primary data storage uses the PostgreSQL `objects` table with JSONB attributes for flexible schema.

```sql
CREATE TABLE objects (
    object_id       SERIAL PRIMARY KEY,
    object_attributes JSONB NOT NULL
);
```

### 3.2 object_attributes JSONB Fields

The `object_attributes` column stores visitor information in JSON format:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Display name of the visitor |
| `visitor_id` | string | Yes | Unique visitor identifier (presence indicates visitor record) |
| `person_name` | string | No | Alternative name field |
| `image_path` | string | No | Path to visitor photo (e.g., `photos/John Doe_123/image.jpg`) |
| `display_name` | string | No | Set to "ignored" for filtered entries |
| `promoted_to_known` | boolean | No | If `true`, entry is protected from ignore operations |

### 3.3 Example JSONB Data

```json
{
    "name": "John Smith",
    "visitor_id": "VS-2024-001",
    "image_path": "photos/John Smith_234/profile.jpg",
    "promoted_to_known": false
}
```

### 3.4 Key PostgreSQL Queries

**Retrieve all visitors:**
```sql
SELECT object_attributes->>'name' as name, object_id
FROM objects
WHERE object_attributes ? 'visitor_id';
```

**Get visitor details by object_id:**
```sql
SELECT object_attributes
FROM objects
WHERE object_id = %s;
```

**Mark entry as ignored:**
```sql
UPDATE objects
SET object_attributes = object_attributes || '{"display_name": "ignored"}'::jsonb
WHERE object_id = %s;
```

**Check if protected (promoted to known):**
```sql
SELECT object_attributes->>'promoted_to_known'
FROM objects
WHERE object_id = %s;
```

### 3.5 Qdrant Vector Database

**Collection**: `visitor_embeddings_office`

| Property | Description |
|----------|-------------|
| **Point ID** | Same as PostgreSQL `object_id` (integer) |
| **Vector** | Face embedding (typically 512 or 768 dimensions) |
| **Payload** | Optional metadata associated with the embedding |
| **Distance Metric** | Cosine similarity (lower threshold = stricter matching) |

---

## 4. Setup Instructions

### 4.1 Prerequisites

Before installation, ensure the following are available:

- [x] Python 3.10+ installed
- [x] PostgreSQL server running with database created
- [x] Qdrant server running (default: http://localhost:6333)
- [x] Face embeddings already populated in Qdrant collection

### 4.2 Environment Setup

**Step 1: Clone or navigate to project directory**
```bash
cd /path/to/visitor-embeddings-dashboard
```

**Step 2: Create virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install fastapi uvicorn psycopg2-binary qdrant-client pyyaml jinja2
```

### 4.3 Dependencies List

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for API and HTML serving |
| `uvicorn` | ASGI server for running FastAPI |
| `psycopg2-binary` | PostgreSQL database adapter |
| `qdrant-client` | Python client for Qdrant vector database |
| `pyyaml` | YAML configuration file parser |
| `jinja2` | Template engine for HTML rendering |

### 4.4 Configuration File Setup

Create or edit `config/config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  dbname: analyticschronicles
  user: postgres
  password: YOUR_PASSWORD_HERE

qdrant:
  url: http://localhost:6333
  collection: visitor_embeddings_office

photos:
  directory: photos
```

### 4.5 Configuration Parameters

| Section | Parameter | Description | Default |
|---------|-----------|-------------|---------|
| `database.host` | PostgreSQL server hostname | `localhost` |
| `database.port` | PostgreSQL server port | `5432` |
| `database.dbname` | Database name | `analyticschronicles` |
| `database.user` | Database username | `postgres` |
| `database.password` | Database password | (required) |
| `qdrant.url` | Qdrant server URL | `http://localhost:6333` |
| `qdrant.collection` | Qdrant collection name | `visitor_embeddings_office` |
| `photos.directory` | Local path to photos folder | `photos` |

### 4.6 Directory Structure

```
project-root/
├── app.py                    # FastAPI application
├── main.py                   # Core functions and utilities
├── config/
│   └── config.yaml          # Configuration file
├── templates/
│   └── dashboard.html       # Jinja2 HTML template
└── photos/                  # Visitor photos directory
    ├── John Doe_123/
    │   └── image.jpg
    ├── unknown/
    │   └── {object_id}/
    └── recognized/
        └── {object_id}/
```

---

## 5. Core Components

### 5.1 main.py Functions

The `main.py` module provides core database and vector search functionality.

#### 5.1.1 Configuration Loading

```python
CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DB_CONFIG = config["database"]
QDRANT_URL = config["qdrant"]["url"]
QDRANT_COLLECTION = config["qdrant"]["collection"]
PHOTOS_DIR = config["photos"]["directory"]
```

#### 5.1.2 Function Reference

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `get_db_connection()` | None | `psycopg2.connection` | Creates PostgreSQL connection using config |
| `get_qdrant_client()` | None | `QdrantClient` | Creates Qdrant client instance |
| `parse_attributes(attrs)` | `attrs: str\|dict` | `dict\|None` | Parses JSONB attributes safely |
| `get_person_name_by_object_id(object_id)` | `object_id: int` | `str\|None` | Retrieves person name from database |
| `get_qdrant_point(point_id)` | `point_id: int` | `Point\|None` | Retrieves single point from Qdrant |
| `get_visitors()` | None | `dict[str, int]` | Returns {name: object_id} for all visitors |
| `find_similar_embeddings(object_id, threshold, limit)` | See below | `list[int]` | Finds similar face embeddings |
| `analyze_user_embeddings(threshold)` | `threshold: float` | None | CLI analysis utility |
| `debug_point_search(point_id)` | `point_id: int` | None | Debug utility for testing thresholds |

#### 5.1.3 find_similar_embeddings Function

```python
def find_similar_embeddings(
    object_id: int,
    similarity_threshold: float = 0.2,
    limit: int = 1000
) -> list[int]:
    """
    Find all Qdrant point IDs similar to the given object_id's embedding.

    Args:
        object_id: The PostgreSQL object_id (same as Qdrant point_id)
        similarity_threshold: Minimum similarity score (0.0 = all, 1.0 = exact)
        limit: Maximum number of results to return

    Returns:
        List of similar object_ids sorted by similarity score
    """
```

### 5.2 app.py Endpoints

The `app.py` module defines the FastAPI application and routes.

#### 5.2.1 Application Setup

```python
app = FastAPI(title="Visitor Embeddings Dashboard")
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")
templates = Jinja2Templates(directory="templates")
```

#### 5.2.2 Helper Functions

| Function | Purpose |
|----------|---------|
| `get_visitor_details(object_id)` | Fetch full visitor attributes from PostgreSQL |
| `get_image_path_for_object(object_id)` | Resolve image path with fallback to unknown/recognized folders |

#### 5.2.3 Image Path Resolution Logic

1. Check `image_path` in `object_attributes`
2. If not found, search `photos/unknown/{object_id}/` for images
3. If still not found, search `photos/recognized/{object_id}/` for images
4. Supports `.jpg` and `.png` formats

### 5.3 dashboard.html Features

#### 5.3.1 UI Components

| Component | Description |
|-----------|-------------|
| **Header** | Title and subtitle with gradient background |
| **Threshold Controls** | Range slider (0-0.5) with Apply button |
| **Visitor Cards** | Expandable cards showing visitor info |
| **Similar Grid** | Scrollable grid of similar face images |
| **Action Bar** | Select All / Mark as Ignored buttons |
| **Toast Notifications** | Success/error/warning messages |

#### 5.3.2 CSS Theme Variables

```css
:root {
    --bg-primary: #0a0e27;      /* Main background */
    --bg-secondary: #121837;    /* Secondary background */
    --bg-card: #1a1f4e;         /* Card background */
    --accent-orange: #ff6b35;   /* Primary accent */
    --accent-purple: #a855f7;   /* Secondary accent */
    --accent-blue: #3b82f6;     /* Tertiary accent */
    --text-primary: #f1f5f9;    /* Main text */
    --text-secondary: #94a3b8;  /* Muted text */
}
```

#### 5.3.3 JavaScript Functions

| Function | Purpose |
|----------|---------|
| `updateThreshold()` | Reload page with new threshold parameter |
| `toggleSimilar(header)` | Expand/collapse similar images section |
| `updateSelection(visitorId)` | Update selected count and button states |
| `toggleSelectAll(btn, visitorId)` | Select/deselect all similar images |
| `ignoreSelected(visitorId)` | Send ignore request to API |
| `showToast(message, type)` | Display notification message |

---

## 6. API Documentation

### 6.1 Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main dashboard HTML page |
| GET | `/api/visitors` | List all visitors |
| GET | `/api/visitor/{object_id}` | Get visitor details with similar embeddings |
| POST | `/api/ignore` | Mark object IDs as ignored |

### 6.2 GET / (Dashboard)

**Description**: Renders the main dashboard HTML page with all visitors and their similar embeddings.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.2 | Similarity threshold for matching (0.0 - 0.5) |

**Example Request**:
```
GET /?threshold=0.15
```

**Response**: HTML page (Content-Type: text/html)

### 6.3 GET /api/visitors

**Description**: Returns a JSON object mapping visitor names to their object IDs.

**Example Request**:
```
GET /api/visitors
```

**Example Response**:
```json
{
    "John Smith": 234,
    "Emily Rodriguez": 555,
    "Michael Chen": 467,
    "Sarah Johnson": 356
}
```

### 6.4 GET /api/visitor/{object_id}

**Description**: Returns detailed information about a specific visitor including similar embedding IDs.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `object_id` | integer | Yes | The visitor's object ID |

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.2 | Similarity threshold |

**Example Request**:
```
GET /api/visitor/234?threshold=0.25
```

**Example Response**:
```json
{
    "object_id": 234,
    "details": {
        "name": "John Smith",
        "visitor_id": "VS-2024-001",
        "image_path": "photos/John Smith_234/profile.jpg",
        "promoted_to_known": false
    },
    "similar_ids": [234, 567, 891, 1023, 1456]
}
```

### 6.5 POST /api/ignore

**Description**: Marks selected object IDs as ignored by updating their `display_name` attribute. Protected entries (with `promoted_to_known: true`) are skipped.

**Request Body**:
```json
{
    "object_ids": [567, 891, 1023]
}
```

**Success Response**:
```json
{
    "success": true,
    "updated": 3
}
```

**Partial Success Response** (some entries protected):
```json
{
    "success": true,
    "updated": 2,
    "skipped": [891],
    "message": "Ignored 2 items. Skipped 1 (promoted to known user)."
}
```

**Error Response**:
```json
{
    "success": false,
    "error": "No object IDs provided"
}
```

### 6.6 Static Files

**Endpoint**: `/photos/*`

**Description**: Serves visitor photos from the configured photos directory.

**Example**: `/photos/John Smith_234/profile.jpg`

---

## 7. User Interface

### 7.1 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│                     HEADER                                   │
│            👤 Visitor Embeddings Dashboard                   │
│         View visitors and manage similar face embeddings     │
├─────────────────────────────────────────────────────────────┤
│                     CONTROLS                                 │
│    Similarity Threshold: [========●===] 0.2  [Apply Filter]  │
├─────────────────────────────────────────────────────────────┤
│                     VISITOR CARDS                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ [👤] John Smith    Object ID: 234    [15 similar] [▼]  │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │ [Select All] [Mark as Ignored]  3 selected            │ │
│  │ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐            │ │
│  │ │☑ 📷│ │☐ 📷│ │☑ 📷│ │☐ 📷│ │☑ 📷│ │☐ 📷│            │ │
│  │ │ID:567│ID:891│ID:1023│ID:1456│ID:1789│ID:2012│        │ │
│  │ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘            │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ [👤] Emily Rodriguez  Object ID: 555  [8 similar] [▼] │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Similarity Threshold Control

The threshold slider controls how similar faces must be to appear in results:

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.0 | Returns all faces (most inclusive) | Broad search for potential matches |
| 0.2 | Default - balanced matching | General usage |
| 0.3 | Stricter matching | Reduce false positives |
| 0.5 | Most strict matching | Only very similar faces |

**Note**: Lower threshold = more results, Higher threshold = fewer but more accurate results

### 7.3 Image Management

#### 7.3.1 Main Visitor Image
- Displayed as circular avatar (70x70px)
- Orange border indicates primary image
- Placeholder icon shown if no image available

#### 7.3.2 Similar Images Grid
- Scrollable grid with max-height of 500px
- Lazy loading for performance
- Image dimensions: 120px height, auto width
- Hover effect with blue border
- Selected items have orange border with glow effect

### 7.4 Ignore Workflow

1. **Expand** a visitor card by clicking the header
2. **Select** similar images using checkboxes (or "Select All")
3. **Click** "Mark as Ignored" button
4. **Observe** toast notification for result
5. **Protected** entries (🔒) remain in grid with yellow border

### 7.5 Visual Indicators

| Indicator | Meaning |
|-----------|---------|
| Orange border on avatar | Primary visitor image |
| Purple badge | Count of similar faces found |
| Blue border on hover | Hoverable image |
| Orange glow on selection | Selected for action |
| Yellow border + 🔒 | Protected (promoted to known user) |
| Green toast | Success message |
| Red toast | Error message |
| Yellow toast | Warning (some items skipped) |

---

## 8. Deployment Process

### 8.1 Development Server

**Quick Start**:
```bash
# Activate virtual environment
source venv/bin/activate

# Run with Uvicorn
python app.py
# OR
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

**Development URL**: http://localhost:8001

### 8.2 Production Deployment

#### 8.2.1 Uvicorn with Multiple Workers

```bash
uvicorn app:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 4 \
    --no-access-log \
    --proxy-headers
```

#### 8.2.2 Systemd Service Configuration

Create `/etc/systemd/system/visitor-dashboard.service`:

```ini
[Unit]
Description=Visitor Embeddings Dashboard
After=network.target postgresql.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/path/to/project
Environment="PATH=/path/to/project/venv/bin"
ExecStart=/path/to/project/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8001 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Enable and start service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable visitor-dashboard
sudo systemctl start visitor-dashboard
sudo systemctl status visitor-dashboard
```

### 8.3 Nginx Reverse Proxy (Optional)

```nginx
server {
    listen 80;
    server_name dashboard.example.com;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /photos/ {
        alias /path/to/project/photos/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }
}
```

### 8.4 Static File Serving

**Development**: FastAPI `StaticFiles` middleware serves `/photos/*` directly.

**Production Considerations**:
- Use Nginx to serve static files for better performance
- Enable caching headers for photos
- Consider CDN for high-traffic deployments
- Ensure proper file permissions (readable by web server user)

### 8.5 Environment Variables (Alternative to config.yaml)

For containerized deployments, consider environment variables:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=analyticschronicles
export DB_USER=postgres
export DB_PASSWORD=secret
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=visitor_embeddings_office
export PHOTOS_DIR=/data/photos
```

### 8.6 Production Checklist

- [ ] Update `config.yaml` with production database credentials
- [ ] Ensure PostgreSQL is accessible from application server
- [ ] Ensure Qdrant is accessible from application server
- [ ] Verify photos directory has correct permissions
- [ ] Configure firewall to allow port 8001 (or use reverse proxy)
- [ ] Set up SSL/TLS certificate for HTTPS
- [ ] Configure log rotation for application logs
- [ ] Set up monitoring and alerting
- [ ] Test all API endpoints after deployment
- [ ] Verify image serving works correctly

---

## 9. Testing & Validation

### 9.1 Prerequisites Verification

**Step 1: Verify PostgreSQL Connection**
```bash
psql -h localhost -U postgres -d analyticschronicles -c "SELECT COUNT(*) FROM objects;"
```

**Step 2: Verify Qdrant Connection**
```bash
curl http://localhost:6333/collections/visitor_embeddings_office
```

**Step 3: Verify Photos Directory**
```bash
ls -la photos/
# Should show visitor folders
```

### 9.2 Functional Testing

#### 9.2.1 Dashboard Load Test
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/
# Expected: 200
```

#### 9.2.2 API Endpoint Tests

**Test /api/visitors**:
```bash
curl http://localhost:8001/api/visitors | python -m json.tool
# Expected: JSON object with visitor names and IDs
```

**Test /api/visitor/{id}**:
```bash
curl "http://localhost:8001/api/visitor/234?threshold=0.2" | python -m json.tool
# Expected: JSON with object_id, details, and similar_ids
```

**Test /api/ignore**:
```bash
curl -X POST http://localhost:8001/api/ignore \
    -H "Content-Type: application/json" \
    -d '{"object_ids": [9999]}' | python -m json.tool
# Expected: {"success": true, "updated": 1} or similar
```

### 9.3 Similarity Search Validation

**Run CLI Analysis**:
```bash
python main.py
```

**Expected Output**:
```
Name                           Object ID    Similar Points
------------------------------------------------------------
John Smith                     234          15
Emily Rodriguez                555          8
...
```

**Debug Specific Point**:
```python
from main import debug_point_search
debug_point_search(234)
```

### 9.4 UI Verification Checklist

- [ ] Dashboard loads without errors
- [ ] All visitor cards display correctly
- [ ] Visitor photos load (or show placeholders)
- [ ] Threshold slider responds to input
- [ ] "Apply Filter" reloads page with new threshold
- [ ] Clicking visitor card expands similar section
- [ ] Similar images load with lazy loading
- [ ] Checkboxes work for selection
- [ ] "Select All" toggles all checkboxes
- [ ] "Mark as Ignored" button enables with selections
- [ ] Ignore action shows appropriate toast message
- [ ] Protected items show lock indicator
- [ ] Ignored items are removed from grid

### 9.5 Error Handling Verification

| Test Case | Expected Behavior |
|-----------|-------------------|
| Invalid object_id in URL | Return empty details |
| Qdrant unavailable | Graceful error, empty similar list |
| PostgreSQL unavailable | 500 error with message |
| Missing photo file | Placeholder displayed |
| Empty ignore request | Error response: "No object IDs provided" |

### 9.6 Performance Validation

**Response Time Benchmarks**:
| Endpoint | Acceptable Time |
|----------|-----------------|
| GET / | < 3 seconds |
| GET /api/visitors | < 500ms |
| GET /api/visitor/{id} | < 1 second |
| POST /api/ignore | < 500ms |

**Load Testing (optional)**:
```bash
# Using Apache Bench
ab -n 100 -c 10 http://localhost:8001/api/visitors
```

---

## Appendix A: Troubleshooting

### Common Issues

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| "Connection refused" to PostgreSQL | Database not running | `sudo systemctl start postgresql` |
| "Connection refused" to Qdrant | Qdrant not running | Start Qdrant service |
| "No module named 'psycopg2'" | Missing dependency | `pip install psycopg2-binary` |
| Photos not displaying | Wrong path in config | Check `photos.directory` setting |
| Empty visitor list | No visitors with `visitor_id` | Verify data in objects table |
| No similar faces found | High threshold or missing embeddings | Lower threshold or check Qdrant data |

### Log Locations

- **Uvicorn**: stdout/stderr (configure with --log-level)
- **Systemd**: `journalctl -u visitor-dashboard -f`
- **Nginx**: `/var/log/nginx/access.log` and `error.log`
