# main.py
from typing import Generator, Optional
import os, re, math, json, threading
import time
import yaml

import cv2
import numpy as np

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    PlainTextResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ========= MediaPipe (face detection + facemesh for eyes gating, no overlay) =========
try:
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection
    mp_fm = mp.solutions.face_mesh
    MP_READY = True
    FM_READY = True
except Exception:
    mp_fd = None
    mp_fm = None
    MP_READY = False
    FM_READY = False

# ========= BioStar Auto-Start (Removed - use config file instead) =========
# The BioStar API configuration is now in config/config_biostar.yaml
# For local testing: Run `python biostar/main.py` separately
# For production: Point biostar_api_base to your production BioStar server

# ========= App =========
app = FastAPI(title="Cortex41 FaceCapture", version="4.3-final")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ========= Config =========
CONFIG_PATH = os.path.abspath(os.path.join("config", "config.json"))
CONFIG_YAML_PATH = os.path.abspath(os.path.join("config", "config.yaml"))

CONFIG_BIOSTAR_PATH = os.path.abspath(os.path.join("config", "config_biostar.yaml"))
CONFIG_LOCK = threading.Lock()
CONFIG: dict = {}
BIOSTAR_CONFIG: dict = {}

CONFIG_MTIME: Optional[float] = None
BIOSTAR_CONFIG_MTIME: Optional[float] = None

LAST_REPORT_ENGINE_URL: Optional[str] = None  # legacy (signup)
LAST_REPORT_ENGINE_BASE_URL: Optional[str] = None
LAST_REPORT_ENGINE_SIGNUP_URL: Optional[str] = None
COMPARE_STATE: dict = {
    # Unix timestamp (float) of the last successful comparison run
    "last_run": None,
    # True when BioStar and Report Engine have exactly the same ID set
    "in_sync": None,
    # IDs present in BioStar but missing in Report Engine
    "missing_ids": [],
    # IDs present in Report Engine but missing in BioStar
    "extra_in_report": [],
    # Full list of IDs from BioStar (as last seen by compare)
    "biostar_ids": [],
    # Full list of IDs from Report Engine (as last seen by compare)
    "report_engine_ids": [],
}

# Last known BioStar session id captured via /biostar_proxy login
BIOSTAR_SESSION_ID: Optional[str] = None


# ========= Compare helpers (BioStar vs Report Engine IDs) =========
def _compare_json_request(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    headers: dict | None = None,
):
    """Internal helper: perform a JSON HTTP request for compare flows.

    Returns (body, response_headers_lowercased).
    """

    import json as _json
    import urllib.error as _urlerr
    import urllib.request as _urlreq

    hdrs: dict = {"Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    data_bytes = None
    if payload is not None:
        data_bytes = _json.dumps(payload).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")

    req = _urlreq.Request(url, data=data_bytes, headers=hdrs, method=method)
    try:
        with _urlreq.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            try:
                body = _json.loads(raw) if raw else {}
            except _json.JSONDecodeError:
                body = {"raw": raw}
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
    except _urlerr.URLError as e:  # Re-raise with a slightly clearer prefix
        raise _urlerr.URLError(f"{url}: {e}") from e

    return body, resp_headers


def _compare_fetch_biostar_ids() -> set[str]:
    """Fetch BioStar user IDs using an existing BioStar session.

    The session id is captured when the user logs into BioStar via the
    /biostar_proxy login endpoint in the dashboard UI. No credentials are
    stored in config or environment variables.
    """

    import urllib.parse as _urlparse

    global BIOSTAR_SESSION_ID

    conf = load_config()
    biostar_conf = conf.get("biostar") or {}

    # Respect biostar.enabled flag - if disabled, do not attempt any calls.
    if not bool(biostar_conf.get("enabled", True)):
        raise RuntimeError(
            "BioStar integration is disabled in config (biostar.enabled=false)"
        )

    biostar_base = str(biostar_conf.get("biostar_api_base") or "").strip()
    if not biostar_base:
        raise RuntimeError(
            "BioStar base URL not configured (biostar.biostar_api_base in config.yaml)"
        )

    endpoints = biostar_conf.get("endpoints") or {}
    users_ep = str(endpoints.get("users_list") or "").strip()
    if not users_ep:
        raise RuntimeError(
            "BioStar users_list endpoint not configured (biostar.endpoints.users_list)"
        )

    session_conf = biostar_conf.get("session") or {}
    session_header_name = (
        str(session_conf.get("header_name") or "bs-session-id").strip()
        or "bs-session-id"
    )

    session_id = BIOSTAR_SESSION_ID
    if not session_id:
        raise RuntimeError(
            "BioStar session is not available. Log into BioStar from the dashboard "
            "before running compare."
        )

    pagination_conf = biostar_conf.get("pagination") or {}
    limit = pagination_conf.get("max_limit") or pagination_conf.get("default_limit") or 100
    try:
        limit = int(limit)
    except Exception:
        limit = 100
    if limit <= 0:
        limit = 100

    ids: set[str] = set()
    offset = 0

    while True:
        base_url = biostar_base.rstrip("/") + users_ep
        qs = _urlparse.urlencode(
            {
                "limit": str(limit),
                "offset": str(offset),
                "last_modified": "0",
            }
        )
        users_url = f"{base_url}?{qs}"
        try:
            print(
                f"[compare] Fetching BioStar users from {users_url} (offset={offset}, limit={limit})...",
                flush=True,
            )
        except Exception:
            pass

        body, _ = _compare_json_request(
            users_url,
            headers={session_header_name: session_id},
        )

        users = []
        total = 0
        if isinstance(body, dict):
            # Try several known BioStar-like response shapes
            if "UserCollection" in body and isinstance(body["UserCollection"], dict):
                coll = body["UserCollection"]
                rows = coll.get("rows") or coll.get("Records") or []
                users = rows if isinstance(rows, list) else []
                total = int(coll.get("total") or coll.get("Total") or 0)
            elif "Data" in body and isinstance(body["Data"], dict):
                data = body["Data"]
                recs = data.get("records") or data.get("rows") or []
                users = recs if isinstance(recs, list) else []
                total = int(data.get("total") or 0)
            elif "data" in body and isinstance(body["data"], dict):
                data = body["data"]
                recs = data.get("records") or data.get("rows") or []
                users = recs if isinstance(recs, list) else []
                total = int(data.get("total") or 0)
            elif "rows" in body and isinstance(body["rows"], list):
                users = body["rows"]
                total = int(body.get("total") or 0)
            elif "records" in body and isinstance(body["records"], list):
                users = body["records"]
                total = int(body.get("total") or 0)
            else:
                # Fallback: if it looks like a collection under some other key
                for v in body.values():
                    if isinstance(v, list):
                        users = v
                        break
        elif isinstance(body, list):
            users = body
            total = len(users)

        if not users:
            break

        for rec in users:
            if not isinstance(rec, dict):
                continue
            uid = rec.get("user_id") or rec.get("UserID") or rec.get("id")
            if uid is None:
                continue
            ids.add(str(uid))

        # Stop condition: either we know total and reached/passed it,
        # or the last page contained fewer than "limit" entries.
        if total > 0 and offset + limit >= total:
            break
        if len(users) < limit:
            break

        offset += limit

    try:
        print(f"[compare] BioStar user IDs: {len(ids)}", flush=True)
    except Exception:
        pass
    return ids


def _compare_fetch_report_engine_ids() -> set[str]:
    """Fetch employee_ids from Report Engine's employees endpoint."""

    base = get_report_engine_base_url().rstrip("/")
    if not base:
        raise RuntimeError(
            "Report Engine base URL not configured (report_engine.base in config.yaml)"
        )

    conf = load_config()
    employees_ep = str(
        _deep_get(conf, "report_engine.endpoints.employees", "") or ""
    ).strip()
    if not employees_ep:
        raise RuntimeError(
            "Report Engine employees endpoint not configured "
            "(report_engine.endpoints.employees in config.yaml)"
        )
    if not employees_ep.startswith("/"):
        employees_ep = "/" + employees_ep

    url = f"{base}{employees_ep}?skip=0&limit=10000"
    try:
        print(f"[compare] Fetching Report Engine employees from {url}...", flush=True)
    except Exception:
        pass

    body, _ = _compare_json_request(url)
    employees = body.get("employees") or []
    ids: set[str] = set()
    for emp in employees:
        if not isinstance(emp, dict):
            continue
        eid = emp.get("employee_id")
        if eid is not None:
            ids.add(str(eid))

    try:
        print(f"[compare] Report Engine employee IDs: {len(ids)}", flush=True)
    except Exception:
        pass
    return ids


def _run_compare_ids_internal() -> dict:
    """Core comparison logic between BioStar and Report Engine IDs.

    This is used by both the /biostar_proxy login path and the
    /compare_run endpoint. It mirrors the old compare_ids.py behaviour
    but lives inside main.py so we don't need a separate container.
    """

    biostar_ids = _compare_fetch_biostar_ids()
    report_ids = _compare_fetch_report_engine_ids()

    missing_in_report = sorted(biostar_ids - report_ids)
    extra_in_report = sorted(report_ids - biostar_ids)
    in_sync = (not missing_in_report) and (not extra_in_report)

    return {
        "biostar_ids": sorted(biostar_ids),
        "report_engine_ids": sorted(report_ids),
        "missing_ids": missing_in_report,
        "extra_in_report": extra_in_report,
        "in_sync": in_sync,
    }


DEFAULT_CONFIG = {
    "roi": {"center_x": 0.5, "center_y": 0.5, "rad_a": 0.21, "rad_b": 0.39},
    "quality": {"area_min": 0.025, "blur_min": 8.0, "jpeg_quality": 80},
    "pose": {
        # slightly tighter symmetric gates: require a bit more yaw to accept side poses
        "yaw_gate_right": 0.07,
        "yaw_gate_left": -0.07,
        "pitch_gate_up": 0.18,
        "yaw_limit_norm": 0.50,
        "yaw_limit_norm_left": 0.50,
        "yaw_limit_norm_right": 0.50
    },
    "eyes": {
        # Horizontal width ratios (visibility proxy)
        "width_thresh": 0.11,
        "width_thresh_left": 0.11,
        "width_thresh_right": 0.11,
        # Vertical open ratio threshold (normalized by eye width)
        "open_thresh": 0.20,
        "open_thresh_left": 0.20,
        "open_thresh_right": 0.20,
        # Which steps require eye gating (left/right require the corresponding eye OPEN)
        "require_steps": ["left", "right"],
        "iris_check": True
    },
    "autocap": {
        "probe_interval_ms": 400, "need_stable": 5,
        "countdown_ticks": 3, "countdown_ms": 700
    },
    "stream": {
        "mesh_overlay": False, "mesh_overlay_stride": 2, "mesh_overlay_style": "full"
    },
    "mirror": {
        "default": True  # UI will start mirrored like a selfie; turn off in production if you like
    }
}

# Fixed BioStar → Report Engine mappings and defaults
# These used to live in config/config.yaml under biostar.field_mappings / biostar.defaults
# but are now hardcoded in code so per‑client config stays simpler.
BIOSTAR_FIELD_MAPPINGS = {
	"user_id": "employee_id",
	"login_id": "password",
	"name": "name",
	"email": "email",
	"gender_map": {"0": "male", "1": "female"},
	"disabled": "is_active",
	"photo": "front_image",
	"birthday": "date_of_birth",
	"address": "address",
	"title": "job_title",
	"phone_number": "phone_number",
	"start_datetime": "date_of_joining",
}

BIOSTAR_DEFAULTS = {
	"role": "user",
	"is_authorized": True,
	"leaves": 16,
	"left_image": None,
	"right_image": None,
	"up_image": None,
}

def _deep_get(d, path, default=None):
    cur = d
    for part in str(path).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def load_config() -> dict:
    """Load unified application config.
    Preference order:
      1) config/config.yaml (YAML, unified)
      2) config/config.json (legacy JSON)
    """
    global CONFIG, CONFIG_MTIME
    with CONFIG_LOCK:
        # Prefer YAML
        try:
            mtime_yaml = os.path.getmtime(CONFIG_YAML_PATH)
        except Exception:
            mtime_yaml = None
        if mtime_yaml is not None:
            if CONFIG_MTIME is not None and CONFIG_MTIME == mtime_yaml and CONFIG:
                return CONFIG
            try:
                with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                # Merge with defaults (defaults overridden by loaded values)
                merged = DEFAULT_CONFIG.copy()
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        merged[k] = v
                CONFIG = merged
                CONFIG_MTIME = mtime_yaml
                return CONFIG
            except Exception as e:
                print(f"[WARNING] Failed to load YAML config: {e}")
                # fall through to JSON
        # Legacy JSON fallback
        try:
            mtime_json = os.path.getmtime(CONFIG_PATH)
        except Exception:
            CONFIG = DEFAULT_CONFIG.copy()
            CONFIG_MTIME = None
            return CONFIG
        if CONFIG_MTIME is not None and CONFIG_MTIME == mtime_json and CONFIG:
            return CONFIG
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            if isinstance(loaded_json, dict):
                for k, v in loaded_json.items():
                    merged[k] = v
            CONFIG = merged
            CONFIG_MTIME = mtime_json
        except Exception as e:
            print(f"[WARNING] Failed to load JSON config: {e}")
            CONFIG = DEFAULT_CONFIG.copy()
            CONFIG_MTIME = None
    return CONFIG

def load_biostar_config() -> dict:
    """Return BioStar configuration.
    Primary source: unified config under key 'biostar'.
    Fallback: legacy config/config_biostar.yaml if present.
    """
    global BIOSTAR_CONFIG, BIOSTAR_CONFIG_MTIME
    # Prefer unified config
    conf = load_config()
    section = conf.get("biostar")
    if isinstance(section, dict) and section:
        merged = dict(section)
        merged.setdefault("field_mappings", BIOSTAR_FIELD_MAPPINGS.copy())
        merged.setdefault("defaults", BIOSTAR_DEFAULTS.copy())
        return merged
    # Fallback to legacy YAML
    with CONFIG_LOCK:
        try:
            mtime = os.path.getmtime(CONFIG_BIOSTAR_PATH)
        except Exception:
            BIOSTAR_CONFIG = {}
            BIOSTAR_CONFIG_MTIME = None
            return BIOSTAR_CONFIG
        if (
            BIOSTAR_CONFIG_MTIME is not None
            and BIOSTAR_CONFIG_MTIME == mtime
            and BIOSTAR_CONFIG
        ):
            return BIOSTAR_CONFIG
        try:
            with open(CONFIG_BIOSTAR_PATH, "r", encoding="utf-8") as f:
                BIOSTAR_CONFIG = yaml.safe_load(f) or {}
            # Ensure fixed mappings/defaults are always present even for legacy file
            if isinstance(BIOSTAR_CONFIG, dict):
                BIOSTAR_CONFIG.setdefault("field_mappings", BIOSTAR_FIELD_MAPPINGS.copy())
                BIOSTAR_CONFIG.setdefault("defaults", BIOSTAR_DEFAULTS.copy())
            BIOSTAR_CONFIG_MTIME = mtime
        except Exception as e:
            print(f"[WARNING] Failed to load legacy BioStar config: {e}")
            BIOSTAR_CONFIG = {}
            BIOSTAR_CONFIG_MTIME = None
    return BIOSTAR_CONFIG

# initial load
load_config()
load_biostar_config()

# ========= Static mounts =========
FACES_DIR = os.path.abspath("faces");  os.makedirs(FACES_DIR, exist_ok=True)
LOGO_DIR  = os.path.abspath("logo");   os.makedirs(LOGO_DIR,  exist_ok=True)
POSES_DIR = os.path.abspath("poses");  os.makedirs(POSES_DIR, exist_ok=True)
TEMPLATES_DIR = os.path.abspath("templates"); os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/faces", StaticFiles(directory=FACES_DIR), name="faces")
app.mount("/logo",  StaticFiles(directory=LOGO_DIR),  name="logo")
app.mount("/poses", StaticFiles(directory=POSES_DIR), name="poses")

# ========= Templates =========
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ========= Frame cache =========
latest_frames: dict[str, "cv2.Mat"] = {}
latest_lock = threading.Lock()

# ========= Helpers =========
def safe_name(s: str) -> str:
	"""Return a filesystem-safe version of an employee name.

	Replaces any non-alphanumeric/underscore/dash characters with underscores and
	ensures we always return a non-empty string (falls back to "employee").
	"""
	return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_") or "employee"


def get_employee_dir(name: str) -> str:
	"""Return (and create) the employee directory under faces/ for a given name.

	Uses safe_name() and also guards against path traversal by normalizing and
	checking the directory remains inside FACES_DIR.
	"""
	emp_safe = safe_name(name)
	base = os.path.abspath(FACES_DIR)
	d = os.path.abspath(os.path.normpath(os.path.join(base, emp_safe)))
	if not d.startswith(base + os.sep):
		raise HTTPException(status_code=400, detail="Invalid employee name")
	os.makedirs(d, exist_ok=True)
	return d

# Date normalization helper for BioStar and manual inputs
from typing import Optional, Any

def normalize_date_yyyy_mm_dd(v: Optional[Any]) -> Optional[str]:
    """Normalize various date formats to YYYY-MM-DD. Return None if invalid.
    Accepts inputs like:
    - YYYY-MM-DDTHH:MM:SS, YYYY-MM-DD HH:MM:SS -> trims to date
    - DD-MM-YYYY, MM-DD-YYYY
    - With '/' instead of '-'
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    # Keep only date part if includes time
    if "T" in s:
        s = s.split("T", 1)[0]
    if " " in s:
        s = s.split(" ", 1)[0]
    s = s.replace("/", "-")
    import datetime as _dt
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d-%m-%y", "%m-%d-%y"):
        try:
            return _dt.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def is_biostar_payload(payload: dict) -> bool:
    """Heuristic to detect BioStar-origin payloads.

    We treat a payload as coming from BioStar if:
      - Email domain contains 'biostar' (e.g., 200669@biostar.local), OR
      - Explicit source flags are present (source=biostar / biostar=true), OR
      - Its employee_id is one of the known BioStar IDs from the last compare.
    """
    try:
        e = str(payload.get("email") or "").lower()
        domain = e.split("@", 1)[1] if "@" in e else ""
        if "biostar" in domain:
            return True
        src = str(payload.get("source") or "").lower()
        if src == "biostar":
            return True
        if bool(payload.get("biostar")):
            return True
        # Fallback: if employee_id is in the last known BioStar ID set, treat
        # it as BioStar-origin even if email/source flags are missing.
        emp_id = str(payload.get("employee_id") or "").strip()
        if emp_id:
            biostar_ids = set(str(x) for x in (COMPARE_STATE.get("biostar_ids") or []))
            if emp_id in biostar_ids:
                return True
    except Exception:
        pass
    return False


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a video source which can be:
    - integer index (0,1,2...)
    - device path (/dev/video0)
    - network URL (rtsp://, http://, https://)

    For RTSP, try FFMPEG backend and prefer TCP transport.
    """
    s = str(source or "0").strip()

    # Numeric index
    try:
        idx = int(s)
        cap = cv2.VideoCapture(idx)
        # Low-latency buffer when possible
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        if not cap or not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Unable to open video source: {source}")
        return cap
    except ValueError:
        pass

    # RTSP handling
    is_rtsp = s.lower().startswith("rtsp://")
    cap = None
    if is_rtsp:
        src = s
        if "rtsp_transport=" not in src:
            sep = "&" if "?" in src else "?"
            src_tcp = f"{src}{sep}rtsp_transport=tcp"
        else:
            src_tcp = src
        # Prefer FFMPEG backend
        try:
            cap = cv2.VideoCapture(src_tcp, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(src_tcp)
        if not cap or not cap.isOpened():
            # Try original URL as fallback
            try:
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            except Exception:
                cap = cv2.VideoCapture(src)
    else:
        # Other URLs or device paths
        cap = cv2.VideoCapture(s)

    # Reduce buffer where supported
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    if not cap or not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Unable to open video source: {source}")
    return cap

from urllib.parse import unquote


def norm_source(src: str) -> str:
	"""Normalize a camera/source string (URL-decode, default to "0")."""
	try:
		return unquote(str(src or "0").strip())
	except Exception:
		return str(src or "0").strip()


def make_source_key(src: str, mirror: bool) -> str:
	"""Make a cache key for latest_frames that encodes source + mirror flag."""
	base = norm_source(src)
	return f"{base}|m" if mirror else base



# ========= Connectivity helper =========
def _normalize_url(raw: str, ensure_trailing_slash: bool = False) -> str:
    """Normalize a URL or host string and optionally enforce a trailing slash."""

    s = str(raw or "").strip()
    if not s:
        return s
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        s = f"http://{s}"
    if ensure_trailing_slash and not s.endswith("/"):
        s += "/"
    return s


def get_report_engine_base_url() -> str:
    """Return the base URL for the Report Engine service.

    Resolution order:
    - REPORT_ENGINE_BASE_URL / REPORT_ENGINE_BASE env vars
    - report_engine_base / report_engine.base from config.yaml
    - Derived from report_engine_signup / report_engine.signup if only that is set.
    """

    global LAST_REPORT_ENGINE_BASE_URL

    raw = (
        os.environ.get("REPORT_ENGINE_BASE_URL")
        or os.environ.get("REPORT_ENGINE_BASE")
        or ""
    )
    raw = str(raw).strip()
    if not raw:
        conf = load_config()
        raw = str(
            _deep_get(conf, "report_engine_base", "")
            or _deep_get(conf, "report_engine.base", "")
        ).strip()
        if not raw:
            # Derive from signup if only that is present
            s = str(
                _deep_get(conf, "report_engine_signup", "")
                or _deep_get(conf, "report_engine.signup", "")
                or _deep_get(conf, "report_engine", "")
            ).strip()
            if s:
                try:
                    import urllib.parse as _urlp

                    p = _urlp.urlparse(_normalize_url(s))
                    raw = f"{p.scheme}://{p.netloc}/"
                except Exception:
                    pass
    if not raw:
        raw = ""
    raw = _normalize_url(raw, ensure_trailing_slash=True)
    if LAST_REPORT_ENGINE_BASE_URL != raw:
        print(f"INFO:     [signup] base set to {raw}", flush=True)
        LAST_REPORT_ENGINE_BASE_URL = raw
    return raw


def get_report_engine_signup_url() -> str:
    """Return the signup endpoint URL used for forwarding employee signups."""

    global LAST_REPORT_ENGINE_SIGNUP_URL

    raw = (
        os.environ.get("REPORT_ENGINE_SIGNUP_URL")
        or os.environ.get("REPORT_ENGINE_URL")
        or os.environ.get("REPORT_ENGINE")
        or ""
    )
    raw = str(raw).strip()
    if not raw:
        conf = load_config()
        raw = str(
            _deep_get(conf, "report_engine_signup", "")
            or _deep_get(conf, "report_engine.signup", "")
            or _deep_get(conf, "report_engine", "")
        ).strip()
    if not raw:
        raw = ""
    raw = _normalize_url(raw, ensure_trailing_slash=False)
    if LAST_REPORT_ENGINE_SIGNUP_URL != raw:
        print(f"INFO:     [signup] signup endpoint set to {raw}", flush=True)
        LAST_REPORT_ENGINE_SIGNUP_URL = raw
    return raw


def get_report_engine_url() -> str:
    """Backward-compatible alias for get_report_engine_signup_url()."""

    return get_report_engine_signup_url()


def is_signup_alive(url: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Check reachability of the Report Engine BASE URL via a lightweight HEAD request."""

    if not url:
        url = get_report_engine_base_url()
    try:
        import urllib.request  # local import to avoid global dependency

        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            return 200 <= int(code) < 400
    except Exception:
        return False


# Background probe that periodically prints status to stdout like server logs
_def_probe_thread = None


def _signup_probe_loop(interval: float = 10.0) -> None:
    """Background loop that periodically logs connectivity status to Report Engine."""

    import time

    # Immediate check then sleep-loop
    while True:
        url = get_report_engine_base_url()
        ok = is_signup_alive(url)
        stamp = time.strftime("%H:%M:%S", time.localtime())
        print(
            f"INFO:     [signup] {stamp} {url} status: {'Connected' if ok else 'No connection'}",
            flush=True,
        )
        time.sleep(max(1.0, float(interval)))

@app.on_event("startup")
async def _start_signup_probe() -> None:
    global _def_probe_thread
    # Start signup probe
    if _def_probe_thread is None:
        _def_probe_thread = threading.Thread(target=_signup_probe_loop, kwargs={"interval": 10.0}, daemon=True)
        _def_probe_thread.start()

# ========= Stream (JPEG quality from config) =========
def mjpeg_generator_with_cache(
	cap: cv2.VideoCapture,
	source_key: str,
	mirror: bool = False,
) -> Generator[bytes, None, None]:
	"""Yield an MJPEG stream and keep the latest frame cached per source.

	Used by /video and pose/capture endpoints so they can reuse the most recent
	frame without reopening the camera.
	"""
	conf = load_config()
	q = int(_deep_get(conf, "quality.jpeg_quality", 80))
	q = 40 if q < 40 else (95 if q > 95 else q)
	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				break
			if mirror:
				frame = cv2.flip(frame, 1)
			with latest_lock:
				latest_frames[source_key] = frame.copy()
			ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
			if not ok:
				continue
			jpg = buf.tobytes()
			yield (
				b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
			)
	finally:
		cap.release()

# ========= RTSP via FFmpeg (safer than OpenCV for some builds) =========
def _iter_jpegs_from_pipe(stdout) -> Generator[bytes, None, None]:
	"""Yield individual JPEG frames from an ffmpeg image2pipe byte stream."""
	buf = bytearray()
	while True:
		chunk = stdout.read(8192)
		if not chunk:
			break
		buf.extend(chunk)
		while True:
			soi = buf.find(b"\xff\xd8")
			if soi < 0:
				break
			eoi = buf.find(b"\xff\xd9", soi + 2)
			if eoi < 0:
				break
			frame = bytes(buf[soi : eoi + 2])
			del buf[: eoi + 2]
			yield frame


def mjpeg_generator_rtsp_ffmpeg(url: str, source_key: str, mirror: bool=False, q: int=80) -> Generator[bytes, None, None]:
    """Uses ffmpeg to read RTSP and outputs MJPEG chunks. Also updates latest_frames cache."""
    if not shutil.which("ffmpeg"):
        print("WARN:     [ffmpeg] binary not found; falling back to OpenCV VideoCapture", flush=True)
        cap = open_capture(url)
        yield from mjpeg_generator_with_cache(cap, source_key, mirror)
        return
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-an", "-f", "image2pipe",
        "-vcodec", "mjpeg", "-q:v", str(5),
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    print(f"INFO:     [ffmpeg] start pid={proc.pid} url={url}", flush=True)
    got_first = False
    try:
        for jpg in _iter_jpegs_from_pipe(proc.stdout):
            if not got_first:
                print("INFO:     [ffmpeg] first frame received", flush=True)
                got_first = True
            # Decode once to keep cache updated for probe/capture endpoints
            try:
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if mirror:
                    frame = cv2.flip(frame, 1)
                with latest_lock:
                    latest_frames[source_key] = frame.copy()
            except Exception as e:
                print(f"WARN:     [ffmpeg] frame decode err: {e}", flush=True)
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        # If we exit the loop, try to log stderr for diagnostics
        try:
            err = proc.stderr.read().decode("utf-8", errors="ignore")
            if err:
                print("WARN:     [ffmpeg] exited; stderr=\n" + err, flush=True)
        except Exception:
            pass
    finally:
        try:
            if proc and proc.poll() is None:
                proc.kill()
        except Exception:
            pass


# ========= Geometry / eyes helpers =========
def _dist(p1, p2) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def _point_in_quad(pt, quad) -> bool:
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    A,B,C,D = quad
    s1 = cross(A,B,pt); s2 = cross(B,C,pt); s3 = cross(C,D,pt); s4 = cross(D,A,pt)
    has_pos = (s1>0)+(s2>0)+(s3>0)+(s4>0)
    has_neg = (s1<0)+(s2<0)+(s3<0)+(s4<0)
    return not (has_pos and has_neg)

def both_eyes_visible_from_mesh(
    mesh_landmarks, W: int, H: int, cfg_eyes: dict
):
    """Analyze FaceMesh landmarks to decide if both eyes are visible and open.

    Returns a dict with per-eye visibility/openness metrics and a combined
    "eyes_ok" flag used by pose/ready gating.
    """

    if mesh_landmarks is None:
        return None

    try:
        lms = mesh_landmarks.landmark
        # Landmark indices (MediaPipe FaceMesh)
        L_OUT, L_IN, L_UP, L_LOW = 33, 133, 159, 145
        R_OUT, R_IN, R_UP, R_LOW = 362, 263, 386, 374
        L_IRIS = [468, 469, 470, 471]
        R_IRIS = [473, 474, 475, 476]

        def px(idx):
            lm = lms[idx]
            return (lm.x * W, lm.y * H)

        # Key points
        pL_out, pL_in, pL_up, pL_low = px(L_OUT), px(L_IN), px(L_UP), px(L_LOW)
        pR_out, pR_in, pR_up, pR_low = px(R_OUT), px(R_IN), px(R_UP), px(R_LOW)

        # Normalizations
        inter_ocular = max(1e-6, _dist(pL_out, pR_out))
        l_width = _dist(pL_out, pL_in)
        r_width = _dist(pR_out, pR_in)

        # Horizontal width ratios (used for visibility)
        l_ratio = l_width / inter_ocular
        r_ratio = r_width / inter_ocular

        # Vertical open ratios normalized by horizontal width (open/closed signal)
        l_open_ratio = _dist(pL_up, pL_low) / max(l_width, 1e-6)
        r_open_ratio = _dist(pR_up, pR_low) / max(r_width, 1e-6)

        # Thresholds
        base_w = float(cfg_eyes.get("width_thresh", 0.13))
        thr_l_w = float(cfg_eyes.get("width_thresh_left", base_w))
        thr_r_w = float(cfg_eyes.get("width_thresh_right", base_w))

        base_o = float(cfg_eyes.get("open_thresh", 0.20))
        thr_l_o = float(cfg_eyes.get("open_thresh_left", base_o))
        thr_r_o = float(cfg_eyes.get("open_thresh_right", base_o))

        left_visible = l_ratio >= thr_l_w
        right_visible = r_ratio >= thr_r_w

        iris_used = False
        if bool(cfg_eyes.get("iris_check", True)) and len(lms) >= 477:
            iris_used = True

            def centroid(idxs):
                xs = [lms[i].x * W for i in idxs]
                ys = [lms[i].y * H for i in idxs]
                return (sum(xs) / len(xs), sum(ys) / len(ys))

            l_iris_c = centroid(L_IRIS)
            r_iris_c = centroid(R_IRIS)
            l_quad = (pL_out, pL_up, pL_in, pL_low)
            r_quad = (pR_out, pR_up, pR_in, pR_low)
            if not _point_in_quad(l_iris_c, l_quad):
                left_visible = False
            if not _point_in_quad(r_iris_c, r_quad):
                right_visible = False

        # An eye is considered "open" only if it's visible AND passes the vertical-open threshold
        left_open = bool(left_visible and (l_open_ratio >= thr_l_o))
        right_open = bool(right_visible and (r_open_ratio >= thr_r_o))

        return {
            "left_ratio": float(l_ratio),
            "right_ratio": float(r_ratio),
            "left_visible": bool(left_visible),
            "right_visible": bool(right_visible),
            "left_open_ratio": float(l_open_ratio),
            "right_open_ratio": float(r_open_ratio),
            "left_open": left_open,
            "right_open": right_open,
            "iris_checks_used": bool(iris_used),
            # For "both eyes" cases, require both to be open
            "eyes_ok": bool(left_open and right_open),
        }
    except Exception:
        return None

# ========= Analysis (ROI-only; FaceDetection + optional FaceMesh) =========
def analyze_with_mediapipe(frame, want: str):
    """
    ROI detection + quality + yaw/pitch + orientation + (eyes via FaceMesh).
    Gate logic:
      - pose gate depends on 'want' (front/right/left/down)
      - if 'want' in eyes.require_steps:
          • for left/right poses require the corresponding eye to be OPEN (not just visible)
          • for other poses (if configured) require BOTH eyes open
      - final ready = inside & close_ok & blur_ok & pose_ok (after eyes)
    """
    conf = load_config()
    if not MP_READY:
        return {"ok": False, "reason": "mp_missing", "details": "MediaPipe not installed."}

    H, W = frame.shape[:2]

    # ROI
    cx = float(_deep_get(conf, "roi.center_x", 0.5))
    cy = float(_deep_get(conf, "roi.center_y", 0.5))
    ra = float(_deep_get(conf, "roi.rad_a", 0.21))
    rb = float(_deep_get(conf, "roi.rad_b", 0.39))

    x0 = max(0, int((cx - ra) * W));  x1 = min(W, int((cx + ra) * W))
    y0 = max(0, int((cy - rb) * H));  y1 = min(H, int((cy + rb) * H))
    if x1 <= x0 or y1 <= y0:
        return {"ok": False, "reason": "roi_invalid"}

    crop = frame[y0:y1, x0:x1]
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Detection on crop
    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.15) as det:
        res = det.process(rgb_crop)

    if not res.detections:
        return {"ok": True, "has_face": False, "ready": False, "inside_oval": False,
                "close_ok": False, "blur_ok": False, "pose_ok": False,
                "eyes_ok": None, "yaw": 0.0, "pitch": 0.0, "orientation": "front"}

    # Largest detection
    def area_of(d):
        bb = d.location_data.relative_bounding_box
        return max(bb.width, 0) * max(bb.height, 0)
    d = max(res.detections, key=area_of)
    bb = d.location_data.relative_bounding_box
    k  = d.location_data.relative_keypoints  # [R_eye, L_eye, nose, mouth, R_ear, L_ear]

    cropW = max(1, x1 - x0)
    cropH = max(1, y1 - y0)
    cx_full = (x0 + (bb.xmin + 0.5 * bb.width) * cropW) / W
    cy_full = (y0 + (bb.ymin + 0.45 * bb.height) * cropH) / H

    # Stricter inside check: require face center AND both eyes inside a slightly tighter ellipse
    inside_scale = float(_deep_get(conf, "roi.inside_scale", 0.95))
    ra_in = ra * inside_scale
    rb_in = rb * inside_scale

    def _in_ellipse(px, py) -> bool:
        return ((px - cx) / ra_in) ** 2 + ((py - cy) / rb_in) ** 2 <= 1.0

    # Convert detection keypoints (relative to crop) to full-frame normalized coords
    r_eye_k, l_eye_k, nose_k = k[0], k[1], k[2]
    r_eye_full = ((x0 + r_eye_k.x * cropW) / W, (y0 + r_eye_k.y * cropH) / H)
    l_eye_full = ((x0 + l_eye_k.x * cropW) / W, (y0 + l_eye_k.y * cropH) / H)

    inside = _in_ellipse(cx_full, cy_full) and _in_ellipse(*r_eye_full) and _in_ellipse(*l_eye_full)

    # quality gates
    area_min = float(_deep_get(conf, "quality.area_min", 0.025))
    blur_min = float(_deep_get(conf, "quality.blur_min", 8.0))
    area_frac = float(bb.width * bb.height) * (cropW * cropH) / (W * H)
    close_ok  = area_frac >= area_min
    blur_val  = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    blur_ok   = blur_val >= blur_min

    # yaw/pitch from keypoints (approx)
    r_eye, l_eye, nose, mouth, r_ear, l_ear = k[0], k[1], k[2], k[3], k[4], k[5]
    dx_r = abs(nose.x - r_ear.x);  dx_l = abs(l_ear.x - nose.x)
    denom = max(dx_r + dx_l, 1e-6)
    yaw = (dx_l - dx_r) / denom               # +right, -left

    eyes_y = (r_eye.y + l_eye.y) * 0.5
    face_h = max(bb.height, 1e-6)
    pitch = (mouth.y - eyes_y) / face_h       # +up

    orientation = "front"
    yaw_right_gate = float(_deep_get(conf, "pose.yaw_gate_right", 0.08))
    yaw_left_gate  = float(_deep_get(conf, "pose.yaw_gate_left", -0.06))
    pitch_up_gate  = float(_deep_get(conf, "pose.pitch_gate_up", 0.18))

    if yaw > yaw_right_gate: orientation = "right"
    elif yaw < yaw_left_gate: orientation = "left"
    if pitch > pitch_up_gate: orientation = "up"  # UI calls this "Down"

    # Pose gates by target
    if   want == "right": pose_ok = yaw >  yaw_right_gate
    elif want == "left":  pose_ok = yaw <  yaw_left_gate
    elif want == "down":  pose_ok = pitch > pitch_up_gate
    else:                 pose_ok = True

    # Eyes via FaceMesh on the ROI crop (server-side only; no overlay)
    eyes_info = None
    if FM_READY:
        try:
            with mp_fm.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            ) as fm:
                fm_res = fm.process(rgb_crop)
            if fm_res.multi_face_landmarks:
                eyes_cfg = conf.get("eyes", {}) if isinstance(conf.get("eyes", {}), dict) else {}
                eyes_info = both_eyes_visible_from_mesh(
                    fm_res.multi_face_landmarks[0],
                    cropW, cropH,
                    eyes_cfg
                )
        except Exception:
            eyes_info = None

    # Eyes gating:
    # - For side poses (left/right), require the corresponding eye to be OPEN
    #   (visible + vertical aperture >= threshold).
    # - For other poses, if configured in eyes.require_steps, require BOTH eyes open.
    req_steps = set(_deep_get(conf, "eyes.require_steps", ["left", "right"]))
    eyes_required = want in req_steps
    eyes_ok = None
    if eyes_required:
        if eyes_info is not None:
            if want == "left":
                eyes_ok = bool(eyes_info.get("left_open", False))
            elif want == "right":
                eyes_ok = bool(eyes_info.get("right_open", False))
            else:
                eyes_ok = bool(eyes_info.get("eyes_ok", False))
            pose_ok = pose_ok and eyes_ok
        else:
            eyes_ok = False
            pose_ok = False
    else:
        eyes_ok = (eyes_info.get("eyes_ok") if eyes_info is not None else None)

    # compute final ready AFTER eyes/pose gating
    has_face = bool(inside)
    ready = has_face and close_ok and blur_ok and pose_ok

    # yaw sanity: if far out-of-bounds AND eyes not verifiably ok, block
    yaw_limit     = float(_deep_get(conf, "pose.yaw_limit_norm", 0.40))
    yaw_limit_l   = float(_deep_get(conf, "pose.yaw_limit_norm_left", yaw_limit))
    yaw_limit_r   = float(_deep_get(conf, "pose.yaw_limit_norm_right", yaw_limit))
    if want in {"left", "right"}:
        limit = yaw_limit_r if want == "right" else yaw_limit_l
        # If yaw is far, allow it only when the eye condition we actually enforce passed.
        eyes_vis_for_limit = eyes_ok if eyes_ok is not None else (eyes_info.get("eyes_ok") if eyes_info else None)
        if abs(yaw) > limit and not bool(eyes_vis_for_limit):
            ready = False
            if eyes_ok is None:
                eyes_ok = False

    out = {
        "ok": True, "has_face": has_face,
        "area_frac": float(area_frac), "blur": float(blur_val),
        "yaw": float(yaw), "pitch": float(pitch),
        "orientation": orientation,
        "inside_oval": bool(inside),
        "close_ok": bool(close_ok), "blur_ok": bool(blur_ok),
        "pose_ok": bool(pose_ok), "ready": bool(ready),
        "eyes_ok": eyes_ok
    }
    if eyes_info is not None:
        out.update({
            # Horizontal width ratios (visibility proxy)
            "left_eye_ratio": eyes_info["left_ratio"],
            "right_eye_ratio": eyes_info["right_ratio"],
            # Vertical open ratios (open/closed proxy)
            "left_eye_open_ratio": eyes_info["left_open_ratio"],
            "right_eye_open_ratio": eyes_info["right_open_ratio"],
            # Per-eye flags
            "left_eye_visible": eyes_info["left_visible"],
            "right_eye_visible": eyes_info["right_visible"],
            "left_eye_open": eyes_info["left_open"],
            "right_eye_open": eyes_info["right_open"],
            "iris_checks_used": eyes_info["iris_checks_used"],
        })
    else:
        out.update({
            "left_eye_ratio": None,
            "right_eye_ratio": None,
            "left_eye_open_ratio": None,
            "right_eye_open_ratio": None,
            "left_eye_visible": None,
            "right_eye_visible": None,
            "left_eye_open": None,
            "right_eye_open": None,
            "iris_checks_used": False
        })
    return out


# ========= Routes =========
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"

@app.get("/signup_status", response_class=JSONResponse)
async def signup_status() -> JSONResponse:
    """Return {'ok': true/false} based on connectivity to configured report engine base URL."""
    return JSONResponse({"ok": bool(is_signup_alive())})


@app.api_route("/biostar_proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def biostar_proxy(path: str, request: Request) -> JSONResponse:
    """Forward HTTP calls from the browser to the configured BioStar API.

    - Solves CORS by letting the browser talk only to FaceCapture.
    - For successful login calls, also triggers a background compare run so
      COMPARE_STATE is fresh before the UI starts auto-sending users.
    """
    import httpx
    import asyncio

    global BIOSTAR_SESSION_ID

    # Get BioStar configuration from unified config
    conf = load_config()
    biostar_conf = conf.get("biostar") or {}

    # Respect biostar.enabled flag; when disabled, treat proxy as unavailable
    if not bool(biostar_conf.get("enabled", True)):
        return JSONResponse(
            {"detail": "BioStar integration is disabled (biostar.enabled=false)"},
            status_code=404,
        )

    biostar_base = str(biostar_conf.get("biostar_api_base") or "").strip()
    if not biostar_base:
        return JSONResponse(
            {"detail": "BioStar base URL not configured (biostar.biostar_api_base)"},
            status_code=500,
        )

    # Build target URL
    target_url = f"{biostar_base.rstrip('/')}/{path}"

    # Get request body if present
    body = None
    try:
        body = await request.body()
    except Exception:
        body = None

    # Forward headers (especially bs-session-id)
    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in ["host", "content-length"]:
            headers[key] = value

    # Make request to BioStar API
    # Determine SSL verification behavior
    verify_param = True
    if str(biostar_base).lower().startswith("https://"):
        # If https and config requests to skip verification, disable SSL verify
        # Default to False (disable) unless explicitly enabled via config under biostar.verify_ssl
        verify_param = bool(_deep_get(conf, "biostar.verify_ssl", False))

    async with httpx.AsyncClient(verify=verify_param) as client:
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params
        )

    # If this was the login endpoint and it succeeded, capture the BioStar
    # session header and run comparison in a background thread. We await it
    # here so that by the time the browser continues (auto-send), main.py
    # already knows which IDs are missing.
    try:
        endpoints = biostar_conf.get("endpoints") or {}
        login_ep = str(
            _deep_get(conf, "biostar.endpoints.login", endpoints.get("login") or "/api/login")
            or "/api/login"
        ).lstrip("/")
        logout_ep = str(
            _deep_get(conf, "biostar.endpoints.logout", endpoints.get("logout") or "/api/logout")
            or "/api/logout"
        ).lstrip("/")

        path_norm = str(path or "").lstrip("/").split("?", 1)[0]

        if path_norm == login_ep and response.status_code == 200:
            # Capture session header from login response
            session_conf = biostar_conf.get("session") or {}
            session_header_name = (
                str(session_conf.get("header_name") or "bs-session-id").strip()
                or "bs-session-id"
            )
            session_id = (
                response.headers.get(session_header_name)
                or response.headers.get(session_header_name.lower())
                or response.headers.get("bs-session-id")
            )
            if session_id:
                BIOSTAR_SESSION_ID = session_id
                try:
                    print(
                        f"INFO:     [biostar_proxy] captured session header {session_header_name} for compare",
                        flush=True,
                    )
                except Exception:
                    pass

            # Reuse the same compare logic as /compare_run, but in a thread so
            # we do not block the event loop on I/O heavy HTTP calls.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _run_compare_ids_internal)
            import time as _time

            COMPARE_STATE["last_run"] = _time.time()
            COMPARE_STATE["in_sync"] = bool(result.get("in_sync"))
            COMPARE_STATE["missing_ids"] = [str(x) for x in result.get("missing_ids") or []]
            COMPARE_STATE["extra_in_report"] = [str(x) for x in result.get("extra_in_report") or []]
            COMPARE_STATE["biostar_ids"] = [str(x) for x in result.get("biostar_ids") or []]
            COMPARE_STATE["report_engine_ids"] = [str(x) for x in result.get("report_engine_ids") or []]

            try:
                print(
                    "INFO:     [compare_sync] (login-triggered) "
                    f"in_sync={COMPARE_STATE['in_sync']} "
                    f"biostar_count={len(COMPARE_STATE['biostar_ids'])} "
                    f"report_engine_count={len(COMPARE_STATE['report_engine_ids'])} "
                    f"missing_in_report={COMPARE_STATE['missing_ids']} "
                    f"extra_in_report={COMPARE_STATE['extra_in_report']}",
                    flush=True,
                )
            except Exception:
                pass
        elif path_norm == logout_ep and response.status_code == 200:
            # Clear stored session when user logs out
            BIOSTAR_SESSION_ID = None
    except Exception as _e:
        try:
            print(f"WARNING: [biostar_proxy] compare failed: {_e}")
        except Exception:
            pass

    # Forward response headers (especially bs-session-id)
    response_headers = {}
    for key, value in response.headers.items():
        # Do NOT forward hop-by-hop headers or headers we override (like content-length)
        if key.lower() not in [
            "content-encoding",
            "content-length",
            "transfer-encoding",
            "connection",
        ]:
            response_headers[key] = value

    # Parse response content safely
    try:
        content = response.json()
    except Exception:
        content = {"data": response.text}

    return JSONResponse(
        content=content,
        status_code=response.status_code,
        headers=response_headers
    )


@app.post("/convert_biostar_image", response_class=JSONResponse)
async def convert_biostar_image(request: Request) -> JSONResponse:
    """
    Convert BioStar base64 PNG image to JPG files and save to disk.
    Accepts payload with 'base64_image' field and 'name' field.
    Saves 4 JPG files: {name}_front.jpg, {name}_left.jpg, {name}_right.jpg, {name}_down.jpg
    Returns the same payload with filenames instead of base64.
    """
    try:
        body = await request.json()
        base64_image = body.get('base64_image')
        employee_name = body.get('name', 'unknown')

        if not base64_image:
            return JSONResponse(
                {"detail": "No base64_image provided"},
                status_code=400
            )

        # Convert base64 PNG to JPG using Pillow
        import base64
        import io
        from PIL import Image

        # Remove data URL prefix if present
        if ',' in base64_image:
            base64_image = base64_image.split(',', 1)[1]

        # Decode base64
        img_bytes = base64.b64decode(base64_image)

        # Open image with PIL
        img = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB if necessary (handles PNG with alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Generate safe name from employee name (keep original case, replace spaces with underscores)
        safe_name = employee_name.replace(' ', '_')
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', safe_name)

        # Create directory for this employee in faces/
        employee_dir = os.path.join('faces', safe_name)
        os.makedirs(employee_dir, exist_ok=True)

        # Save 4 copies with different pose names
        poses = ['front', 'left', 'right', 'down']
        filenames = {}

        for pose in poses:
            # Use hyphen separator to match existing pattern (e.g., John_Smith-front.jpg)
            filename = f"{safe_name}-{pose}.jpg"
            filepath = os.path.join(employee_dir, filename)

            # Save JPG to disk
            img.save(filepath, format='JPEG', quality=85)

            # Store just the filename (not full path) for response
            filenames[pose] = filename

            print(f"[INFO] Saved {filepath}")

        # Remove base64_image from payload and add filenames
        body.pop('base64_image', None)
        body['front_image'] = filenames['front']
        body['left_image'] = filenames['left']
        body['right_image'] = filenames['right']
        body['up_image'] = filenames['down']

        print(f"[INFO] Converted payload for {employee_name}:")
        print(f"  - front_image: {body.get('front_image')}")
        print(f"  - left_image: {body.get('left_image')}")
        print(f"  - right_image: {body.get('right_image')}")
        print(f"  - up_image: {body.get('up_image')}")

        return JSONResponse(content=body, status_code=200)

    except Exception as e:
        return JSONResponse(
            {"detail": f"Image conversion error: {str(e)}"},
            status_code=500
        )


@app.post("/signup_forward", response_class=JSONResponse)
async def signup_forward(request: Request) -> JSONResponse:
    """
    Forward to Report Engine signup endpoint.
    - If target host is host.docker.internal (local dev), send JSON (application/json)
    - Otherwise, send multipart/form-data with form fields (to match production Form/File signature).
    - If image filenames are present and files exist under faces/, attach them as file parts.
    """
    url = get_report_engine_signup_url()
    # Post directly to the canonical path (no trailing slash) to avoid 307 redirect
    post_url = url.rstrip('/')
    try:
        body_bytes = await request.body()
        body_str = (body_bytes or b"{}").decode("utf-8", errors="ignore")
        try:
            j = json.loads(body_str) if body_str else {}
        except Exception:
            j = {}
        imgs = [j.get('front_image'), j.get('left_image'), j.get('right_image'), j.get('up_image')]
        print(f"INFO:     [signup_forward] forwarding to {post_url} name={j.get('name')} email={j.get('email')} employee_id={j.get('employee_id')} images={imgs}")

        import httpx, urllib.parse, os
        target = urllib.parse.urlparse(post_url)
        host = (target.netloc or "").lower()
        is_local_json = False  # always use multipart/form-data for parity with production

        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            if not is_local_json:
                # Build multipart form data for production
                data: dict[str, str] = {}
                def add(k, v, default=None):
                    val = v if v is not None else default
                    if val is None:
                        return
                    if isinstance(val, bool):
                        data[k] = "true" if val else "false"
                    else:
                        data[k] = str(val)
                # Required
                add("email", j.get("email"))
                # Password policy
                # Force a strong default for all cases to satisfy upstream policy
                pwd_final = "OmniClouds@2026!"
                try:
                    print("INFO:     [signup_forward] forcing password override for upstream policy")
                except Exception:
                    pass
                add("password", pwd_final)
                add("name", j.get("name"))
                # Optionals (with defaults aligned to manual flow)
                add("role", j.get("role"), "user")
                add("is_active", j.get("is_active"), True)
                add("is_admin", j.get("is_admin"), False)
                add("is_authorized", j.get("is_authorized"), False)
                add("employee_id", j.get("employee_id"))
                add("gender", j.get("gender"))
                # Normalize date_of_birth
                _dob = normalize_date_yyyy_mm_dd(j.get("date_of_birth"))
                if _dob:
                    add("date_of_birth", _dob)
                elif j.get("date_of_birth"):
                    try:
                        print("INFO:     [signup_forward] dropping invalid date_of_birth (expected YYYY-MM-DD)")
                    except Exception:
                        pass
                add("address", j.get("address"))
                add("job_title", j.get("job_title"))
                add("employment_type", j.get("employment_type"))
                add("work_location", j.get("work_location"))
                add("work_schedule", j.get("work_schedule"))
                add("department", j.get("department"))
                add("phone_number", j.get("phone_number"))
                # Normalize date_of_joining to YYYY-MM-DD; drop if invalid
                _doj = normalize_date_yyyy_mm_dd(j.get("date_of_joining"))
                if _doj:
                    add("date_of_joining", _doj)
                elif j.get("date_of_joining"):
                    try:
                        print("INFO:     [signup_forward] dropping invalid date_of_joining (expected YYYY-MM-DD)")
                    except Exception:
                        pass
                add("leaves", j.get("leaves"), 16)
                add("facility_id", j.get("facility_id"))

                # Attach image files when available
                files = {}
                open_files = []
                def resolve_file(fname: Optional[str]) -> Optional[str]:
                    if not fname:
                        return None
                    f = str(fname)
                    # Absolute/URL-path cases
                    if f.startswith('/faces/'):
                        p = os.path.abspath(f.lstrip('/'))
                        return p if os.path.exists(p) else None
                    if f.startswith('faces/'):
                        p = os.path.abspath(f)
                        return p if os.path.exists(p) else None
                    if os.path.isabs(f):
                        return f if os.path.exists(f) else None
                    # Bare filename: try multiple candidate dirs
                    raw_name = j.get('name') or 'employee'
                    # Default dir logic used by app
                    try:
                        emp_dir_default = get_employee_dir(raw_name)
                    except Exception:
                        emp_dir_default = os.path.abspath('faces')
                    # Dir logic used by convert_biostar_image
                    import re as _re
                    alt_safe = _re.sub(r'[^a-zA-Z0-9_]', '', str(raw_name).replace(' ', '_')) or 'employee'
                    emp_dir_alt = os.path.abspath(os.path.join('faces', alt_safe))
                    candidates = [
                        os.path.join(emp_dir_default, f),
                        os.path.join(emp_dir_alt, f),
                        os.path.abspath(os.path.join('faces', f)),
                    ]
                    for cand in candidates:
                        if os.path.exists(cand):
                            return cand
                    return None

                # Determine if this payload comes from BioStar (so we can
                # apply compare-IDs gating logic only to synced flows).
                is_biostar = is_biostar_payload(j)

                # If we have compare state and this looks like a BioStar user,
                # only send when employee_id is one of the "missing" IDs. If
                # in_sync is True, we skip sending entirely.
                emp_id = str(j.get("employee_id") or "").strip()
                if is_biostar:
                    in_sync = bool(COMPARE_STATE.get("in_sync"))
                    missing_ids = set(str(x) for x in (COMPARE_STATE.get("missing_ids") or []))
                    if in_sync:
                        try:
                            print(
                                f"INFO:     [signup_forward] skipping send for BioStar user {emp_id} "
                                "because compare_ids reports in_sync",
                                flush=True,
                            )
                        except Exception:
                            pass
                        return JSONResponse(
                            {
                                "status": "skipped",
                                "reason": "ids_in_sync",
                                "employee_id": emp_id,
                            },
                            status_code=200,
                        )
                    if emp_id and missing_ids and emp_id not in missing_ids:
                        try:
                            print(
                                f"INFO:     [signup_forward] skipping send for BioStar user {emp_id} "
                                "because employee_id is not in missing_ids",
                                flush=True,
                            )
                        except Exception:
                            pass
                        return JSONResponse(
                            {
                                "status": "skipped",
                                "reason": "not_missing_in_report_engine",
                                "employee_id": emp_id,
                            },
                            status_code=200,
                        )

                for field in ("front_image", "left_image", "right_image", "up_image"):
                    p = resolve_file(j.get(field))
                    if p:
                        try:
                            fobj = open(p, "rb")
                            open_files.append(fobj)
                            files[field] = (os.path.basename(p), fobj, "image/jpeg")
                        except Exception:
                            pass

                # Debug: show exactly which keys we are sending
                try:
                    print(f"INFO:     [signup_forward] data_keys={list(data.keys())} files_keys={list(files.keys())} -> {post_url}")
                except Exception:
                    pass

                # Force multipart/form-data even when no files (to satisfy Form+File signature)
                enforced_files = files if files else {"__dummy": ("", b"", "application/octet-stream")}
                resp = await client.post(post_url, data=data, files=enforced_files, headers={"Accept": "application/json"})
                code = int(resp.status_code)
                for f in open_files:
                    try:
                        f.close()
                    except Exception:
                        pass
            else:



                # Local dev server expects JSON
                resp = await client.post(post_url, content=(body_bytes or b"{}"), headers={"Content-Type": "application/json", "Accept": "application/json"})
                code = int(resp.status_code)

            try:
                data_out = resp.json()
            except Exception:
                data_out = {"detail": (resp.text or "")}
            if 200 <= code < 300:
                return JSONResponse(content=data_out, status_code=code)
            else:
                print(f"INFO:     [signup_forward] upstream error code={code} body={data_out}")
                return JSONResponse(content=data_out, status_code=code)
    except Exception as ex:
        return JSONResponse({"detail": f"forward_error: {ex}"}, status_code=502)



@app.get("/compare_state", response_class=JSONResponse)
async def get_compare_state() -> JSONResponse:
    """Return the last in-memory comparison result (COMPARE_STATE).

    The frontend uses this to mark which BioStar users are already present
    in the Report Engine DB and to display basic sync status.
    """
    from copy import deepcopy

    state = deepcopy(COMPARE_STATE)
    return JSONResponse(state)


@app.post("/compare_sync_signal", response_class=JSONResponse)
async def compare_sync_signal(request: Request) -> JSONResponse:
    """Accept a comparison summary from an external tool.

    The compare_ids.py CLI (or any other tool) can POST a payload with
    in_sync/missing/extra/id lists. We normalize and store it in
    COMPARE_STATE so other endpoints and the UI can reuse the result.
    """

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    in_sync = bool(payload.get("in_sync"))
    missing_ids_raw = payload.get("missing_ids") or []
    extra_in_report_raw = payload.get("extra_in_report") or []
    biostar_ids_raw = payload.get("biostar_ids") or []
    report_ids_raw = payload.get("report_engine_ids") or []

    missing_ids = [str(x) for x in missing_ids_raw if x is not None]
    extra_in_report = [str(x) for x in extra_in_report_raw if x is not None]
    biostar_ids = [str(x) for x in biostar_ids_raw if x is not None]
    report_ids = [str(x) for x in report_ids_raw if x is not None]

    biostar_count = int(payload.get("biostar_count") or 0)
    report_count = int(payload.get("report_engine_count") or 0)

    # Update simple in-memory state so we can inspect later if needed.
    try:
        import time as _time

        COMPARE_STATE["last_run"] = _time.time()
    except Exception:
        COMPARE_STATE["last_run"] = None

    COMPARE_STATE["in_sync"] = in_sync
    COMPARE_STATE["missing_ids"] = missing_ids
    COMPARE_STATE["extra_in_report"] = extra_in_report
    COMPARE_STATE["biostar_ids"] = biostar_ids
    COMPARE_STATE["report_engine_ids"] = report_ids

    # Log a concise summary for troubleshooting.
    try:
        if in_sync:
            print(
                "INFO:     [compare_sync] in_sync=True "
                f"biostar_count={biostar_count} report_engine_count={report_count}",
                flush=True,
            )
        else:
            print(
                "INFO:     [compare_sync] in_sync=False "
                f"biostar_count={biostar_count} report_engine_count={report_count} "
                f"missing_in_report={missing_ids} extra_in_report={extra_in_report}",
                flush=True,
            )
    except Exception:
        pass

    return JSONResponse(
        {
            "status": "ok",
            "in_sync": in_sync,
            "missing_ids": missing_ids,
            "extra_in_report": extra_in_report,
            "biostar_ids": biostar_ids,
            "report_engine_ids": report_ids,
            "biostar_count": biostar_count,
            "report_engine_count": report_count,
        }
    )



@app.post("/compare_run", response_class=JSONResponse)
async def compare_run() -> JSONResponse:
    """Run a fresh BioStar vs Report Engine ID comparison now.

    This uses the same internal comparison helper that the /biostar_proxy
    login path uses, updates COMPARE_STATE, and returns the summary to
    the caller.
    """
    import asyncio
    import time as _time

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _run_compare_ids_internal)
    except Exception as ex:
        return JSONResponse({"detail": f"compare_error: {ex}"}, status_code=500)

    missing_ids = [str(x) for x in (result.get("missing_ids") or []) if x is not None]
    extra_in_report = [str(x) for x in (result.get("extra_in_report") or []) if x is not None]
    biostar_ids = [str(x) for x in (result.get("biostar_ids") or []) if x is not None]
    report_ids = [str(x) for x in (result.get("report_engine_ids") or []) if x is not None]
    in_sync = bool(result.get("in_sync"))

    COMPARE_STATE["last_run"] = _time.time()
    COMPARE_STATE["in_sync"] = in_sync
    COMPARE_STATE["missing_ids"] = missing_ids
    COMPARE_STATE["extra_in_report"] = extra_in_report
    COMPARE_STATE["biostar_ids"] = biostar_ids
    COMPARE_STATE["report_engine_ids"] = report_ids

    try:
        if in_sync:
            print(
                "INFO:     [compare_run] in_sync=True "
                f"biostar_count={len(biostar_ids)} report_engine_count={len(report_ids)}",
                flush=True,
            )
        else:
            print(
                "INFO:     [compare_run] in_sync=False "
                f"biostar_count={len(biostar_ids)} report_engine_count={len(report_ids)} "
                f"missing_in_report={missing_ids} extra_in_report={extra_in_report}",
                flush=True,
            )
    except Exception:
        pass

    return JSONResponse(
        {
            "status": "ok",
            "in_sync": in_sync,
            "missing_ids": missing_ids,
            "extra_in_report": extra_in_report,
            "biostar_ids": biostar_ids,
            "report_engine_ids": report_ids,
            "biostar_count": len(biostar_ids),
            "report_engine_count": len(report_ids),
        }
    )





@app.get("/config", response_class=JSONResponse)
async def get_config() -> JSONResponse:
    conf = load_config()
    biostar_conf = load_biostar_config()

    return JSONResponse({
        "autocap": {
            "probe_interval_ms": int(_deep_get(conf, "autocap.probe_interval_ms", 400)),
            "need_stable": int(_deep_get(conf, "autocap.need_stable", 5)),
            "countdown_ticks": int(_deep_get(conf, "autocap.countdown_ticks", 3)),
            "countdown_ms": int(_deep_get(conf, "autocap.countdown_ms", 700)),
        },
        "mirror": {
            "default": bool(_deep_get(conf, "mirror.default", True))
        },
        "report_engine_signup": str(_deep_get(conf, "report_engine_signup", _deep_get(conf, "report_engine.signup", ""))),
        "biostar": biostar_conf  # Send entire BioStar config to frontend
    })

@app.get("/video")
async def video(request: Request, src: Optional[str] = "0", mirror: Optional[bool] = False):
    """Serve MJPEG stream; also caches latest (mirrored if requested) for probes/captures."""
    source = norm_source(src)
    source_key = make_source_key(source, bool(mirror))
    # Prefer FFmpeg path for RTSP for stability
    if source.lower().startswith("rtsp://"):
        print(f"INFO:     [video] RTSP via ffmpeg selected for {source}", flush=True)
        gen = mjpeg_generator_rtsp_ffmpeg(source, source_key, mirror=bool(mirror))
        return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")
    # Fallback: OpenCV capture (index, device path, or file/URL)
    cap = open_capture(source)
    return StreamingResponse(
        mjpeg_generator_with_cache(cap, source_key, mirror=bool(mirror)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/pose_probe", response_class=JSONResponse)
async def pose_probe(src: Optional[str] = "0", target: Optional[str] = "front", mirror: Optional[bool] = False):
    """Poll at cadence; returns ready state for current target. Uses mirrored frame if mirror=true."""
    source_key = make_source_key(src or "0", bool(mirror))
    with latest_lock:
        frame = latest_frames.get(source_key)

    if frame is None:
        return JSONResponse({"ready": False, "reason": "no_frame", "has_face": False})

    if target not in {"front", "left", "right", "down"}:
        target = "front"

    result = analyze_with_mediapipe(frame, target)
    if not result.get("ok", False):
        return JSONResponse({
            "ready": False,
            "reason": result.get("reason", "unknown"),
            "details": result.get("details", ""),
            "has_face": False,
            "inside": False,
        })

    return JSONResponse({
        "ready": bool(result["ready"]),
        "orientation": result["orientation"],
        "inside": bool(result["inside_oval"]),
        "close_ok": bool(result["close_ok"]),
        "blur_ok": bool(result["blur_ok"]),
        "pose_ok": bool(result["pose_ok"]),
        "eyes_ok": result.get("eyes_ok", None),
        # Eye metrics (visibility + openness)
        "left_eye_ratio": result.get("left_eye_ratio"),
        "right_eye_ratio": result.get("right_eye_ratio"),
        "left_eye_open_ratio": result.get("left_eye_open_ratio"),
        "right_eye_open_ratio": result.get("right_eye_open_ratio"),
        "left_eye_visible": result.get("left_eye_visible"),
        "right_eye_visible": result.get("right_eye_visible"),
        "left_eye_open": result.get("left_eye_open"),
        "right_eye_open": result.get("right_eye_open"),
        "yaw": result["yaw"], "pitch": result["pitch"],
        "has_face": bool(result["has_face"]),
    })

@app.post("/capture_labeled", response_class=JSONResponse)
async def capture_labeled(
    src: Optional[str] = "0",
    label: Optional[str] = "front",
    name: str = "",
    email: str = "",
    mirror: Optional[bool] = False
):
    """
    Capture only if all conditions green (inside, close_ok, blur_ok, pose_ok, eyes_ok if needed).
    Uses the same mirrored frame if mirror=true.
    """
    if label not in {"front", "left", "right", "down"}:
        raise HTTPException(status_code=400, detail="label must be one of ['front','left','right','down']")

    source_key = make_source_key(src or "0", bool(mirror))
    with latest_lock:
        frame = latest_frames.get(source_key)
    if frame is None:
        raise HTTPException(status_code=400, detail="No frame available yet. Start the stream first.")

    analysis = analyze_with_mediapipe(frame, label)

    if not (analysis.get("ok") and analysis.get("ready")):
        reason = {
            "has_face": analysis.get("has_face"),
            "inside": analysis.get("inside_oval"),
            "close_ok": analysis.get("close_ok"),
            "blur_ok": analysis.get("blur_ok"),
            "pose_ok": analysis.get("pose_ok"),
            "eyes_ok": analysis.get("eyes_ok"),
            "left_eye_ratio": analysis.get("left_eye_ratio"),
            "right_eye_ratio": analysis.get("right_eye_ratio"),
            "yaw": analysis.get("yaw"),
            "pitch": analysis.get("pitch"),
            "orientation": analysis.get("orientation"),
        }
        raise HTTPException(status_code=409, detail={"message": "Not ready for capture", "reason": reason})

    employee_dir = get_employee_dir(name or "employee")
    filename = f"{safe_name(name) if name else 'employee'}-{label}.jpg"
    path = os.path.join(employee_dir, filename)

    ok = cv2.imwrite(path, frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save image")

    url = f"/faces/{safe_name(name or 'employee')}/{filename}"
    return JSONResponse({
        "message": f"Saved {label} for {name or 'employee'} at {url}",
        "filename": filename,
        "label": label,
        "url": url,
        "path": path,
        "analysis": analysis,
    })

# ========= Entrypoint =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=True)
