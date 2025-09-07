# main.py
from __future__ import annotations

import os
import re
import json
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, time, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from dateutil.parser import parse as parse_dt
import pdfplumber

# LangChain Groq LLM (for PDF extraction)
from langchain_groq import ChatGroq
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    from langchain.schema import HumanMessage, SystemMessage  # type: ignore

# Your LLM service (must be available in PYTHONPATH)
from llm_service import LLMService

# ----------------------------- Load env -----------------------------
load_dotenv()

# ----------------------------- Flask & CORS --------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ----------------------------- Mongo Setup --------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "hackathon")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
resources_collection = db[os.getenv("RESOURCES_COLLECTION", "resources")]
projects_collection  = db[os.getenv("PROJECTS_COLLECTION", "projects")]
allocation_collection = db[os.getenv("ALLOCATION_COLLECTION", "allocation")]

# ----------------------------- LLM Service --------------------------
llm_service = LLMService()

# ----------------------------- Upload dir ---------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# ================== JSON helpers (ObjectId, datetime -> strings) ==================
def _to_serializable(x):
    if isinstance(x, datetime):
        # Return ISO with Z if it has tzinfo=UTC
        if x.tzinfo:
            return x.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return x.isoformat()
    if isinstance(x, ObjectId):
        return str(x)
    return x

def _coerce_doc(doc: dict) -> dict:
    return {k: _to_serializable(v) for k, v in (doc or {}).items()}

def _coerce_many(cursor):
    return [_coerce_doc(d) for d in cursor]

# ================== PDF → Projects Extractor (LLM) ==================
MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 7000
CHUNK_OVERLAP = 500
MAX_CHUNKS = 24

VALID_STATUSES = {"Not Started", "In Progress", "Completed", "On Hold"}
STATUS_ALIASES = {
    "planned": "Not Started", "notstarted": "Not Started",
    "inprogress": "In Progress", "ongoing": "In Progress", "working": "In Progress",
    "onhold": "On Hold", "paused": "On Hold",
    "done": "Completed", "complete": "Completed", "completed": "Completed",
}

CHUNK_JSON_SPEC = """
You extract ALL project-like items from the given document CHUNK and return a STRICT JSON ARRAY.
Each array item must match this schema:

{
  "name": string,
  "description": string|null,
  "required_skills": string[],
  "start_date": string|null,
  "end_date": string|null,
  "budget": number,
  "resources": string[],
  "status": "Not Started"|"In Progress"|"Completed"|"On Hold"
}

Rules:
- Output ONLY a JSON array, NO commentary.
- Defaults: description=null, required_skills=[], start_date=null, end_date=null, budget=0, resources=[], status="Not Started".
- Dates MUST be YYYY-MM-DD if you can infer, else null.
- If no projects, return [].
"""

def read_pdf_text(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        parts = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(parts).strip()

def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, max_chunks=MAX_CHUNKS):
    s = s or ""
    chunks, start = [], 0
    while start < len(s) and len(chunks) < max_chunks:
        end = min(len(s), start + size)
        chunks.append(s[start:end])
        if end == len(s):
            break
        start = max(0, end - overlap)
    return chunks

def _to_utc_midnight(dt_like: date | datetime) -> datetime:
    if isinstance(dt_like, datetime):
        d = dt_like.date()
    else:
        d = dt_like
    return datetime.combine(d, time.min, tzinfo=timezone.utc)

def _mongo_iso(dt_obj: Optional[date | datetime]) -> Optional[str]:
    if dt_obj is None:
        return None
    if isinstance(dt_obj, date) and not isinstance(dt_obj, datetime):
        dt_obj = _to_utc_midnight(dt_obj)
    elif isinstance(dt_obj, datetime):
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        else:
            dt_obj = dt_obj.astimezone(timezone.utc)
    return dt_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def to_date_or_none(v) -> Optional[date]:
    if not v or str(v).lower() in ("—", "null", "none"):
        return None
    try:
        return parse_dt(str(v), fuzzy=True).date()
    except Exception:
        return None

def to_float_or_zero(v) -> float:
    if not v or str(v).lower() in ("—", "null", "none"):
        return 0.0
    try:
        return float(v)
    except Exception:
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
        return float(nums[0]) if nums else 0.0

def to_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [p.strip() for p in re.split(r"[;,\n,]+", str(v)) if p.strip()]

def norm_status(s: Optional[str]) -> str:
    if not s or str(s).strip() == "":
        return "Not Started"
    s2 = str(s).strip()
    if s2 in VALID_STATUSES:
        return s2
    key = "".join(ch for ch in s2.lower() if ch.isalnum())
    return STATUS_ALIASES.get(key, "Not Started")

def normalize_project(obj: dict) -> dict:
    start_d = to_date_or_none(obj.get("start_date"))
    end_d = to_date_or_none(obj.get("end_date"))
    return {
        "name": str(obj.get("name") or "").strip(),
        "description": (obj.get("description") if obj.get("description") not in ("", "—") else None),
        "required_skills": to_list(obj.get("required_skills")),
        "start_date": _mongo_iso(start_d),
        "end_date": _mongo_iso(end_d),
        "budget": to_float_or_zero(obj.get("budget")),
        "resources": to_list(obj.get("resources")),
        "status": norm_status(obj.get("status")),
    }

def llm_extract_chunk(chunk: str, file_label: str) -> List[dict]:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set in .env")

    prompt = f"""{CHUNK_JSON_SPEC}

FILE: {file_label}

DOCUMENT CHUNK:
\"\"\"{chunk}\"\"\""""
    llm = ChatGroq(model=MODEL_NAME, temperature=0)
    msgs = [SystemMessage(content="You output strict JSON only."), HumanMessage(content=prompt)]
    out = llm.invoke(msgs)
    raw = out.content if hasattr(out, "content") else str(out)

    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        arr = json.loads(raw[start:end+1])
        if not isinstance(arr, list):
            return []
    except Exception:
        return []
    return [normalize_project(item) for item in arr]

def merge_projects(candidates: List[dict]) -> List[dict]:
    if not candidates:
        return []
    def key(n): return re.sub(r"\s+", " ", (n or "").strip().lower())
    grouped: Dict[str, List[dict]] = {}
    for p in candidates:
        grouped.setdefault(key(p.get("name")), []).append(p)

    STATUS_PRIORITY = ["Completed", "In Progress", "On Hold", "Not Started"]
    rank = {s: i for i, s in enumerate(STATUS_PRIORITY)}
    from collections import Counter

    merged = []
    for _, items in grouped.items():
        names = [it["name"] for it in items if it["name"]]
        name = Counter(names).most_common(1)[0][0] if names else ""
        descs = [it["description"] for it in items if it.get("description")]
        description = max(descs, key=len) if descs else None

        def dedupe(seq):
            seen, out = set(), []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out

        skills = dedupe([s for it in items for s in it.get("required_skills", [])])
        resources = dedupe([r for it in items for r in it.get("resources", [])])

        starts = [it["start_date"] for it in items if it.get("start_date")]
        ends = [it["end_date"] for it in items if it.get("end_date")]

        def parse_iso_z(s):
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        start_dt = min(map(parse_iso_z, starts)) if starts else None
        end_dt = max(map(parse_iso_z, ends)) if ends else None

        budgets = [it["budget"] for it in items if it.get("budget", 0) > 0]
        budget = max(budgets) if budgets else 0.0

        statuses = [it["status"] for it in items if it.get("status")]
        status = sorted(statuses, key=lambda s: rank.get(s, 999))[0] if statuses else "Not Started"

        if name:
            merged.append({
                "name": name,
                "description": description,
                "required_skills": skills,
                "start_date": _mongo_iso(start_dt),
                "end_date": _mongo_iso(end_dt),
                "budget": float(budget),
                "resources": resources,
                "status": status,
                "import_date": _mongo_iso(datetime.now(timezone.utc)),
            })
    return merged

def extract_all_projects_from_pdf(input_file: str) -> List[dict]:
    text = read_pdf_text(input_file)
    if not text.strip():
        return []
    chunks = chunk_text(text)
    all_candidates = []
    for ch in chunks:
        all_candidates.extend(llm_extract_chunk(ch, Path(input_file).name))
    return merge_projects(all_candidates)

# ================== PDF → Resources Extractor (LLM) ==================
RESOURCE_JSON_SPEC = """
You extract ALL resource-like items from the given document CHUNK and return a STRICT JSON ARRAY.
Each array item must match this schema:

{
  "name": string,
  "role": string|null,
  "skills": string[],
  "proficiency": "intern"|"junior"|"mid"|"senior"|null,
  "location": string|null,
  "rate_per_hour": number|null,
  "availability_start": string|null
}

Rules:
- Output ONLY a JSON array, NO commentary.
- Defaults: role=null, skills=[], proficiency=null, location=null, rate_per_hour=null, availability_start=null.
- Dates MUST be YYYY-MM-DD if you can infer, else null.
- If you can't find a field, use null.
- If no resources, return [].
"""

PROF_LEVELS = {"intern", "junior", "mid", "senior"}
PROF_ALIASES = {
    "internship": "intern",
    "jr": "junior",
    "jr.": "junior",
    "sr": "senior",
    "sr.": "senior",
    "middle": "mid",
    "mid-level": "mid",
    "midlevel": "mid",
}

def to_float_or_none(v) -> Optional[float]:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except Exception:
        m = re.findall(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
        return float(m[0]) if m else None

def to_dt_or_none(v) -> Optional[datetime]:
    if v is None or str(v).strip() == "":
        return None
    try:
        d = parse_dt(str(v), fuzzy=True)
        # Normalize to UTC midnight when only date is provided
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        else:
            d = d.astimezone(timezone.utc)
        # If no time provided, use midnight
        return d
    except Exception:
        return None

def norm_prof(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = str(s).strip().lower()
    s2 = PROF_ALIASES.get(s2, s2)
    return s2 if s2 in PROF_LEVELS else None

def normalize_resource(obj: dict) -> dict:
    return {
        "name": str(obj.get("name") or "").strip(),
        "role": (obj.get("role") or None),
        "skills": to_list(obj.get("skills")),
        "proficiency": norm_prof(obj.get("proficiency")),
        "location": (obj.get("location") or None),
        "rate_per_hour": to_float_or_none(obj.get("rate_per_hour")),
        "availability_start": _mongo_iso(to_dt_or_none(obj.get("availability_start"))),
    }

def llm_extract_resources_chunk(chunk: str, file_label: str) -> List[dict]:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set in .env")

    prompt = f"""{RESOURCE_JSON_SPEC}

FILE: {file_label}

DOCUMENT CHUNK:
\"\"\"{chunk}\"\"\""""
    llm = ChatGroq(model=MODEL_NAME, temperature=0)
    msgs = [SystemMessage(content="You output strict JSON only."), HumanMessage(content=prompt)]
    out = llm.invoke(msgs)
    raw = out.content if hasattr(out, "content") else str(out)

    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        arr = json.loads(raw[start:end+1])
        if not isinstance(arr, list):
            return []
    except Exception:
        return []
    return [normalize_resource(item) for item in arr]

def merge_resources(candidates: List[dict]) -> List[dict]:
    """Merge by name; union skills; prefer highest proficiency; earliest availability; lowest rate."""
    if not candidates:
        return []

    def key(n): return re.sub(r"\s+", " ", (n or "").strip().lower())
    grouped: Dict[str, List[dict]] = {}
    for r in candidates:
        nm = r.get("name")
        if not nm:
            continue
        grouped.setdefault(key(nm), []).append(r)

    prof_rank = {"intern": 0, "junior": 1, "mid": 2, "senior": 3}

    merged = []
    for _, items in grouped.items():
        # name: pick the most common non-empty
        from collections import Counter
        names = [it["name"] for it in items if it.get("name")]
        if not names:
            continue
        name = Counter(names).most_common(1)[0][0]

        # role: take longest non-null as "most informative"
        roles = [it.get("role") for it in items if it.get("role")]
        role = max(roles, key=len) if roles else None

        # skills: union + stable order
        def dedupe(seq):
            seen, out = set(), []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out
        skills = dedupe([s for it in items for s in (it.get("skills") or [])])

        # proficiency: pick highest rank
        profs = [it.get("proficiency") for it in items if it.get("proficiency") in prof_rank]
        proficiency = max(profs, key=lambda x: prof_rank[x]) if profs else None

        # availability_start: earliest
        def parse_iso_z(s):
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        avails = [it.get("availability_start") for it in items if it.get("availability_start")]
        avail_dt = min(map(parse_iso_z, avails)) if avails else None

        # rate: lowest
        rates = [it.get("rate_per_hour") for it in items if isinstance(it.get("rate_per_hour"), (int, float))]
        rate = min(rates) if rates else None

        # location: most common non-empty
        locs = [it.get("location") for it in items if it.get("location")]
        location = Counter(locs).most_common(1)[0][0] if locs else None

        merged.append({
            "name": name,
            "role": role,
            "skills": skills,
            "proficiency": proficiency,
            "location": location,
            "rate_per_hour": rate,
            "availability_start": _mongo_iso(avail_dt) if avail_dt else None,
            "import_date": _mongo_iso(datetime.now(timezone.utc)),
        })
    return merged

def extract_all_resources_from_pdf(input_file: str) -> List[dict]:
    text = read_pdf_text(input_file)
    if not text.strip():
        return []
    chunks = chunk_text(text)
    all_candidates = []
    for ch in chunks:
        all_candidates.extend(llm_extract_resources_chunk(ch, Path(input_file).name))
    return merge_resources(all_candidates)


# ================== Project persistence helpers ==================
def update_or_upsert_projects(projects: List[dict], upsert: bool = False) -> Dict[str, int]:
    """
    Update projects by 'name'. If upsert=True, insert missing.
    Returns counts dict.
    """
    counters = {"updated": 0, "inserted": 0, "unchanged": 0, "not_found": 0}
    for p in projects:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        if upsert:
            res = projects_collection.update_one({"name": name}, {"$set": p}, upsert=True)
            if res.upserted_id:
                counters["inserted"] += 1
            elif res.modified_count:
                counters["updated"] += 1
            else:
                counters["unchanged"] += 1
        else:
            # update only if exists
            exists = projects_collection.find_one({"name": name})
            if not exists:
                counters["not_found"] += 1
                continue
            res = projects_collection.update_one({"_id": exists["_id"]}, {"$set": p})
            if res.modified_count:
                counters["updated"] += 1
            else:
                counters["unchanged"] += 1
    return counters
def update_or_upsert_resources(resources: List[dict], upsert: bool = False) -> Dict[str, int]:
    """
    Update resources by 'name'. If upsert=True, insert when missing.
    Returns counts dict.
    """
    counters = {"updated": 0, "inserted": 0, "unchanged": 0, "not_found": 0}
    for r in resources:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        if upsert:
            res = resources_collection.update_one({"name": name}, {"$set": r}, upsert=True)
            if res.upserted_id:
                counters["inserted"] += 1
            elif res.modified_count:
                counters["updated"] += 1
            else:
                counters["unchanged"] += 1
        else:
            exists = resources_collection.find_one({"name": name})
            if not exists:
                counters["not_found"] += 1
                continue
            res = resources_collection.update_one({"_id": exists["_id"]}, {"$set": r})
            if res.modified_count:
                counters["updated"] += 1
            else:
                counters["unchanged"] += 1
    return counters


def refresh_llm_state():
    """
    Refresh the in-memory resources/projects inside LLMService
    after DB changes so the chat suggestions reflect the latest data.
    """
    try:
        resources, projects = llm_service.load_data_from_mongo()
        st = llm_service.state
        st["resources"] = resources
        st["projects"] = projects
        # Clear ranked/pending because underlying data changed
        st["ranked"] = []
        st["pending_candidate"] = None
    except Exception as e:
        print(f"⚠️ Failed to refresh LLM state: {e}")

# ================== Routes ==================
@app.get("/")
def home():
    return "Backend is running!", 200

@app.post("/chat")
def chat_endpoint():
    payload = request.get_json(silent=True) or {}
    user_msg = (payload.get("message") or "").strip()
    print("POST /chat payload:", payload, flush=True)

    if not user_msg:
        return jsonify({
            "text": "(empty message)",
            "projects": [], "resources": [], "allocation": [],
            "gantt": [], "heatmap": []
        }), 200

    try:
        data = llm_service.ask(user_msg)  # {text, projects, resources, allocation}
        data["projects"] = [_coerce_doc(p) for p in (data.get("projects") or [])]
        data["resources"] = [_coerce_doc(r) for r in (data.get("resources") or [])]
        data["allocation"] = [_coerce_doc(a) for a in (data.get("allocation") or [])]

        # Optional small demo charts if backend didn't compute any
        data.setdefault("gantt", [
            {"name": "Project Alpha", "hours": 20},
            {"name": "Project Beta", "hours": 35},
        ])
        data.setdefault("heatmap", [
            {"skill": "AI", "usage": 80},
            {"skill": "Full-Stack", "usage": 60},
        ])
        print("POST /chat response keys:", list(data.keys()), flush=True)
        return jsonify(data), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "text": f"Server error: {e}",
            "projects": [], "resources": [], "allocation": [],
            "gantt": [], "heatmap": []
        }), 200

@app.get("/catalog")
def catalog():
    try:
        projects = _coerce_many(projects_collection.find({}))
        resources = _coerce_many(resources_collection.find({}))
        return jsonify({"projects": projects, "resources": resources}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "projects": [], "resources": []}), 200

@app.get("/assignments")
def list_assignments():
    try:
        rows = _coerce_many(allocation_collection.find({}))
        return jsonify(rows), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify([]), 200

@app.post("/allocate")
def allocate():
    """
    Body:
    {
      "project_id": "<ObjectId or string>",
      "resource_id": "<ObjectId or string>",
      "fit_score": 45,
      "reason": "Matched skills: React, Node.js",
      "allocated_on": "2025-09-06T18:21:52.431Z"  # optional
    }
    """
    try:
        body = request.get_json(silent=True) or {}
        proj_id = body.get("project_id")
        res_id = body.get("resource_id")

        def _to_oid(val):
            return ObjectId(val) if val and not isinstance(val, ObjectId) else val

        # Parse ISO including 'Z'
        def _parse_isoz(s):
            if not s:
                return datetime.utcnow()
            s2 = s.replace("Z", "+00:00")
            return datetime.fromisoformat(s2)

        doc = {
            "project_id": _to_oid(proj_id),
            "resource_id": _to_oid(res_id),
            "fit_score": body.get("fit_score"),
            "reason": body.get("reason"),
            "allocated_on": _parse_isoz(body.get("allocated_on")),
            "__v": int(body.get("__v", 0))
        }

        if not doc["project_id"] or not doc["resource_id"]:
            return jsonify({"status": "error", "message": "project_id and resource_id are required"}), 400

        result = allocation_collection.insert_one(doc)
        saved = allocation_collection.find_one({"_id": result.inserted_id})
        return jsonify({"status": "success", "data": _coerce_doc(saved)}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 200

@app.post("/uploadprojectfile")
def upload_project_file():
    """
    Multipart: field 'file' must be .pdf.
    Default: UPDATE existing Mongo 'projects' by name (no inserts).
    Add ?persist=true to UPSERT (insert if missing).
    Refreshes LLMService in-memory state after DB writes.
    Returns JSON summary.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only .pdf allowed"}), 400

        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        # Extract projects from PDF via Groq LLM
        projects = extract_all_projects_from_pdf(path)

        # Persist
        do_upsert = str(request.args.get("persist", "")).lower() in ("1", "true", "yes")
        counters = {"updated": 0, "inserted": 0, "unchanged": 0, "not_found": 0}
        if projects:
            counters = update_or_upsert_projects(projects, upsert=do_upsert)
            refresh_llm_state()  # keep chat in sync

        return jsonify({
            "filename": filename,
            "project_count": len(projects),
            **counters,
            "projects": projects
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.post("/upload_resource_file")
def upload_resource_file():
    """
    Multipart: field 'file' must be .pdf.
    Default: UPDATE existing Mongo 'resources' by name (no inserts).
    Add ?persist=true to UPSERT (insert if missing).
    Refreshes LLMService in-memory state after DB writes.
    Returns JSON summary.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only .pdf allowed"}), 400

        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        # Extract resources from PDF via Groq LLM
        resources = extract_all_resources_from_pdf(path)

        # Persist
        do_upsert = str(request.args.get("persist", "")).lower() in ("1", "true", "yes")
        counters = {"updated": 0, "inserted": 0, "unchanged": 0, "not_found": 0}
        if resources:
            counters = update_or_upsert_resources(resources, upsert=do_upsert)
            refresh_llm_state()  # keep chat in sync

        return jsonify({
            "filename": filename,
            "resource_count": len(resources),
            **counters,
            "resources": resources
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({"ok": True}), 200

# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    # Use port 5000 (match your React/Node calls)
    app.run(host="0.0.0.0", port=8000, debug=False)
