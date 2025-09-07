from __future__ import annotations

from typing import Dict, List, Any, Optional, TypedDict, Tuple
from datetime import datetime
from dotenv import load_dotenv
import os, re, json, difflib

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pymongo import MongoClient
try:
    from bson import ObjectId  # type: ignore
except Exception:
    ObjectId = type("ObjectIdStub", (), {})

LLM_OK = True


class LLMService:
    class State(TypedDict):
        chat_history: List[BaseMessage]
        display_log: List[Dict[str, str]]
        project: Optional[Dict[str, Any]]
        last_project_name: Optional[str]
        ranked: List[Dict[str, Any]]
        rejected: Dict[str, List[str]]
        confirmed: List[Dict[str, str]]
        awaiting: bool
        preferences_by_project: Dict[str, Dict[str, Any]]
        resources: List[Dict[str, Any]]
        projects: List[Dict[str, Any]]
        pending_candidate: Optional[Dict[str, Any]]

    CONFIRM_WORDS = r"(accept|confirm|allocate|assign|allot|alot|select|choose|finalize)"
    PRONOUNS = r"(him|her|them|this|that|above|candidate|person|one)"

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        resources_collection: Optional[str] = None,
        projects_collection: Optional[str] = None,
        allocation_collection: Optional[str] = None,
        llm_model: str = "llama-3.1-8b-instant",
    ):
        load_dotenv()

        self.MONGO_URI = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.MONGO_DB = db_name or os.getenv("MONGO_DB", "hackathon")
        self.RESOURCES_COLLECTION = resources_collection or os.getenv("RESOURCES_COLLECTION", "resources")
        self.PROJECTS_COLLECTION = projects_collection or os.getenv("PROJECTS_COLLECTION", "projects")
        self.ALLOCATION_COLLECTION = allocation_collection or os.getenv("ALLOCATION_COLLECTION", "allocation")

        self.llm_model = llm_model

        self._mongo_client: Optional[MongoClient] = None
        self._db = None
        self._res_col = None
        self._proj_col = None
        self._alloc_col = None

        self._init_mongo()
        self.state: LLMService.State = self.new_state()

    # -------------------- Mongo helpers --------------------

    def _init_mongo(self):
        self._mongo_client = MongoClient(self.MONGO_URI)
        self._db = self._mongo_client[self.MONGO_DB]
        self._res_col = self._db[self.RESOURCES_COLLECTION]
        self._proj_col = self._db[self.PROJECTS_COLLECTION]
        self._alloc_col = self._db[self.ALLOCATION_COLLECTION]

    @staticmethod
    def _coerce_list(v) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [v]
        return [str(v)]

    @staticmethod
    def _coerce_float(v) -> Optional[float]:
        try:
            return float(v) if v is not None and v != "" else None
        except Exception:
            return None

    @staticmethod
    def _json_default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        try:
            if isinstance(o, ObjectId):  # type: ignore
                return str(o)
        except Exception:
            pass
        return str(o)

    def _jsonify(self, v: Any) -> Any:
        if isinstance(v, datetime):
            return v.isoformat()
        try:
            if isinstance(v, ObjectId):  # type: ignore
                return str(v)
        except Exception:
            pass
        if isinstance(v, dict):
            return {k: self._jsonify(val) for k, val in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [self._jsonify(x) for x in v]
        return v

    def _clean_resource(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        avail = doc.get("availability_start") or ""
        if isinstance(avail, datetime):
            avail = avail.isoformat()
        return {
            "name": doc.get("name", ""),
            "role": doc.get("role", ""),
            "skills": self._coerce_list(doc.get("skills")),
            "proficiency": (doc.get("proficiency") or "").lower(),
            "location": doc.get("location", ""),
            "rate_per_hour": self._coerce_float(doc.get("rate_per_hour")),
            "availability_start": avail,
        }

    def _clean_project(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        start = doc.get("start_date") or ""
        if isinstance(start, datetime):
            start = start.isoformat()
        return {
            "name": doc.get("name", ""),
            "required_skills": self._coerce_list(doc.get("required_skills")),
            "start_date": start,
            "location": doc.get("location", ""),
            "status": doc.get("status", ""),
        }

    def load_data_from_mongo(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_resources = list(self._res_col.find({}, {"_id": 0}))
        raw_projects = list(self._proj_col.find({}, {"_id": 0}))
        resources = [self._clean_resource(r) for r in raw_resources]
        projects = [self._clean_project(p) for p in raw_projects]
        return resources, projects

    def _coerce_doc_for_json(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return self._jsonify(doc)

    def build_llm_json(
        self,
        fetch_projects: bool = True,
        fetch_resources: bool = True,
        fetch_allocation: bool = True
    ) -> Tuple[str, str, str]:
        projects = (
            [self._coerce_doc_for_json(p) for p in self._proj_col.find({}, {"_id": 0})]
            if fetch_projects else []
        )
        resources = (
            [self._coerce_doc_for_json(r) for r in self._res_col.find({}, {"_id": 0})]
            if fetch_resources else []
        )
        allocation_suggestions = (
            [self._coerce_doc_for_json(a) for a in self._alloc_col.find({}, {"_id": 0})]
            if fetch_allocation else []
        )

        projects_json = json.dumps(projects, indent=2, ensure_ascii=False, default=self._json_default) if projects else ""
        resources_json = json.dumps(resources, indent=2, ensure_ascii=False, default=self._json_default) if resources else ""
        allocation_json = json.dumps(allocation_suggestions, indent=2, ensure_ascii=False, default=self._json_default) if allocation_suggestions else ""
        return projects_json, resources_json, allocation_json

    # -------------------- Matching helpers --------------------

    @staticmethod
    def _parse_iso(s: Optional[str]) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
        except Exception:
            return None

    @staticmethod
    def _norm(xs: List[str]) -> List[str]:
        return [x.lower().strip() for x in xs or []]

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = s.lower()
        s = re.sub(r'[^a-z0-9]+', ' ', s)
        return re.sub(r'\s+', ' ', s).strip()

    def _skill_match(self, req: List[str], have: List[str]) -> float:
        r, h = self._norm(req), self._norm(have)
        return len(set(r) & set(h)) / len(r) if r else 0.0

    def _availability(self, pstart, ravail) -> float:
        p, r = self._parse_iso(pstart), self._parse_iso(ravail)
        if not p or not r:
            return 0.5
        if r <= p:
            return 1.0
        d = (r - p).days
        if d <= 7:
            return 0.8
        if d <= 14:
            return 0.6
        if d <= 30:
            return 0.4
        return 0.1

    def _base_score(self, project, res) -> float:
        s = self._skill_match(project.get("required_skills", []), res.get("skills", []))
        a = self._availability(project.get("start_date"), res.get("availability_start"))
        loc_bonus = 0.05 if project.get("location", "").lower() == res.get("location", "").lower() else 0.0
        return s * 0.65 + a * 0.3 + loc_bonus

    def _find_project(self, projects: List[Dict[str, Any]], name: str):
        if not name:
            return None
        q_raw = name.strip().lower()
        q_norm = self._normalize_text(name)
        for p in projects:
            if (p.get("name") or "").lower() == q_raw:
                return p
        names_norm = []
        for p in projects:
            pn = p.get("name") or ""
            pn_norm = self._normalize_text(pn)
            names_norm.append((p, pn.lower(), pn_norm))
            if pn_norm == q_norm or q_norm in pn_norm or pn_norm in q_norm:
                return p
        originals = [p.get("name", "").lower() for p in projects]
        best = difflib.get_close_matches(q_raw, originals, n=1, cutoff=0.6)
        if best:
            for p in projects:
                if p.get("name", "").lower() == best[0]:
                    return p
        norm_list = [pn for _, _, pn in names_norm]
        bestn = difflib.get_close_matches(q_norm, norm_list, n=1, cutoff=0.6)
        if bestn:
            idx = norm_list.index(bestn[0])
            return names_norm[idx][0]
        return None

    def detect_project_from_text(self, projects: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
        t = text.lower()
        for p in projects:
            name = (p.get("name") or "").lower()
            if not name:
                continue
            if name in t:
                return p
            toks = [w for w in re.split(r'\W+', name) if w]
            if len(toks) >= 2:
                bigrams = [' '.join(toks[i:i + 2]) for i in range(len(toks) - 1)]
                if any(bg in t for bg in bigrams):
                    return p
            uniq = set(toks)
            if uniq:
                found = sum(1 for w in uniq if w in t)
                if found / len(uniq) >= 0.6:
                    return p
        return None

    @staticmethod
    def _find_resource(resources: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        q = name.lower().strip()
        for r in resources:
            if r.get("name", "").lower() == q:
                return r
        for r in resources:
            if q in r.get("name", "").lower():
                return r
        return None

    @staticmethod
    def extract_skills(t: str) -> List[str]:
        t = t.lower()
        SKILLS = [
            "python", "react", "node", "node.js", "nlp", "ml", "cv", "sql", "tensorflow",
            "flask", "mongodb", "c++", "web3", "solidity", "dl", "or-tools", "data visualization"
        ]
        SYN = {"computer vision": "cv", "machine learning": "ml", "nodejs": "node.js", "reactjs": "react"}
        hits: List[str] = []
        for phrase, norm in SYN.items():
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", t):
                if norm not in hits:
                    hits.append(norm)
        for sw in SKILLS:
            if re.search(rf"(?<!\w){re.escape(sw)}(?!\w)", t):
                if sw not in hits:
                    hits.append(sw)
        return hits

    def parse_preferences(self, user_text: str) -> Dict[str, Any]:
        t = user_text.lower()
        prefs: Dict[str, Any] = {"hard": {}, "soft": {}, "skills": []}
        m_loc = re.search(r"(?:who.?s\s+)?location\s+is\s+in\s+([a-z\s]+)|(?:in|at|located in)\s+([a-z\s]+)", t)
        if m_loc:
            loc = (m_loc.group(1) or m_loc.group(2) or "").strip()
            if "prefer" in t:
                prefs["soft"]["location"] = loc
            else:
                prefs["hard"]["location"] = loc
        for prof in ["senior", "mid", "intern", "junior"]:
            if re.search(rf"\b{prof}\b", t):
                hard = any(w in t for w in ["need", "must", "only", "strict", "require"])
                (prefs["hard"] if hard else prefs["soft"])["proficiency"] = prof
        m_rate = re.search(r"(?:rate|cost|price).*?(?:<=|<|under|below)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", t) or \
                 re.search(r"(?:<=|<|under|below)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/hr|per hour)?", t)
        if m_rate:
            try:
                prefs["hard"]["max_rate"] = float(m_rate.group(1))
            except Exception:
                pass
        prefs["skills"] = self.extract_skills(t)
        hard_words = r"(must|need|required|only|strict)"
        hard_sk = []
        # FIX: don't call the list
        for sw in prefs["skills"]:
            pat = rf"(?<!\w){re.escape(sw)}(?!\w)"
            if re.search(rf"{hard_words}.{{0,12}}{pat}|{pat}.{{0,12}}{hard_words}", t):
                hard_sk.append(sw)
        if hard_sk:
            prefs.setdefault("hard", {})["skills"] = hard_sk
        if re.search(r"(lowest|cheapest|less rate|low rate|min(imum)? rate)", t):
            prefs["mode"] = "lowest_rate"
        if re.search(r"(highest|max(imum)? rate|more rate|most expensive)", t):
            prefs["mode"] = "highest_rate"
        return prefs

    def apply_preferences_score(self, project: Dict[str, Any], res: Dict[str, Any], prefs: Dict[str, Any]) -> Optional[float]:
        hard = prefs.get("hard", {})
        if hard.get("location") and res.get("location", "").lower() != hard["location"]:
            return None
        if hard.get("proficiency") and res.get("proficiency", "").lower() != hard["proficiency"]:
            return None
        if hard.get("max_rate") is not None:
            rate = res.get("rate_per_hour")
            if rate is not None and rate > hard["max_rate"]:
                return None
        hard_sk = set(hard.get("skills", []))
        if hard_sk:
            res_sk = set(self._norm(res.get("skills", [])))
            if not hard_sk.issubset(res_sk):
                return None
        score = self._base_score(project, res)
        soft = prefs.get("soft", {})
        if soft.get("location") and res.get("location", "").lower() == soft["location"]:
            score += 0.05
        if soft.get("proficiency") and res.get("proficiency", "").lower() == soft["proficiency"]:
            score += 0.03
        for sw in prefs.get("skills", []):
            if sw in self._norm(res.get("skills", [])):
                score += 0.02
        return round(score, 3)

    # -------------------- Ranking / state --------------------

    def new_state(self) -> State:
        resources, projects = self.load_data_from_mongo()
        return {
            "chat_history": [],
            "display_log": [],
            "project": None,
            "last_project_name": None,
            "ranked": [],
            "rejected": {},
            "confirmed": [],
            "awaiting": False,
            "preferences_by_project": {},
            "resources": resources,
            "projects": projects,
            "pending_candidate": None,
        }

    @staticmethod
    def _merge_by_name(items: List[Dict[str, Any]], new_item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        name = (new_item.get("name") or "").lower()
        if not name:
            return items, "skipped"
        updated = False
        out = []
        for it in items:
            if (it.get("name") or "").lower() == name:
                merged = it.copy()
                merged.update(new_item)
                out.append(merged)
                updated = True
            else:
                out.append(it)
        if not updated:
            out.append(new_item)
            return out, "added"
        return out, "updated"

    def ingest_json_payload(self, state: State, obj: Any) -> str:
        changed = []
        if isinstance(obj, dict) and ("resources" in obj or "projects" in obj):
            if "resources" in obj and isinstance(obj["resources"], list):
                for r in obj["resources"]:
                    state["resources"], _ = self._merge_by_name(state["resources"], self._clean_resource(r))
                changed.append(f"resources(+{len(obj['resources'])})")
            if "projects" in obj and isinstance(obj["projects"], list):
                for p in obj["projects"]:
                    state["projects"], _ = self._merge_by_name(state["projects"], self._clean_project(p))
                changed.append(f"projects(+{len(obj['projects'])})")
        elif isinstance(obj, dict):
            if "role" in obj or ("skills" in obj and "availability_start" in obj):
                state["resources"], act = self._merge_by_name(state["resources"], self._clean_resource(obj))
                changed.append(f"resource({act})")
            elif "required_skills" in obj or "start_date" in obj:
                state["projects"], act = self._merge_by_name(state["projects"], self._clean_project(obj))
                changed.append(f"project({act})")
            else:
                return "Unrecognized object; expected resource or project fields."
        elif isinstance(obj, list):
            r_like = sum(1 for x in obj if isinstance(x, dict) and ("role" in x or "availability_start" in x))
            p_like = sum(1 for x in obj if isinstance(x, dict) and ("required_skills" in x or "start_date" in x))
            if r_like >= p_like:
                for r in obj:
                    state["resources"], _ = self._merge_by_name(state["resources"], self._clean_resource(r))
                changed.append(f"resources(+{len(obj)})")
            else:
                for p in obj:
                    state["projects"], _ = self._merge_by_name(state["projects"], self._clean_project(p))
                changed.append(f"projects(+{len(obj)})")
        else:
            return "Unsupported JSON payload."
        if state["project"]:
            pname = state["project"]["name"]
            prefs = state["preferences_by_project"].get(pname, {"hard": {}, "soft": {}, "skills": []})
            rejected = state["rejected"].get(pname, [])
            state["ranked"] = self.rank_for_project(state["project"], rejected, prefs, state)
            state["pending_candidate"] = {"resource": state["ranked"][0]["resource"], "project": pname} if state["ranked"] else None
        return "Updated: " + ", ".join(changed) if changed else "Nothing changed."

    def rank_for_project(self, project: Dict[str, Any], rejected: List[str], prefs: Dict[str, Any], state: State) -> List[Dict[str, Any]]:
        mode = prefs.get("mode")

        def _qualified(rsrc: Dict[str, Any]) -> bool:
            req = set(self._norm(project.get("required_skills", [])))
            res_sk = set(self._norm(rsrc.get("skills", [])))
            return req.issubset(res_sk)

        def _rate(x: Dict[str, Any]) -> float:
            rt = x.get("rate_per_hour")
            return float(rt) if isinstance(rt, (int, float)) else 1e9

        if mode in ("lowest_rate", "highest_rate"):
            candidates: List[Tuple[Dict[str, Any], float]] = []
            for r in state["resources"]:
                if r.get("name") in rejected:
                    continue
                if self.apply_preferences_score(project, r, prefs) is None:
                    continue
                if not _qualified(r):
                    continue
                candidates.append((r, self._base_score(project, r)))
            if not candidates:
                return []
            if mode == "lowest_rate":
                candidates.sort(key=lambda pair: (_rate(pair[0]), -pair[1]))
            else:
                candidates.sort(key=lambda pair: (-_rate(pair[0]), -pair[1]))
            return [{"resource": r, "score": round(sc, 3)} for r, sc in candidates[:10]]

        ranked: List[Dict[str, Any]] = []
        for r in state["resources"]:
            if r.get("name") in rejected:
                continue
            adj = self.apply_preferences_score(project, r, prefs)
            if adj is None:
                continue
            if adj >= 0.35:
                ranked.append({"resource": r, "score": adj})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        if not ranked:
            # FIX: don't call the list
            for rr in state["resources"]:
                if rr.get("name") in rejected:
                    continue
                adj2 = self.apply_preferences_score(project, rr, prefs)
                if adj2 is None:
                    continue
                if adj2 >= 0.30:
                    ranked.append({"resource": rr, "score": round(adj2, 3)})
            ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    @staticmethod
    def top(state: State) -> Optional[Dict[str, Any]]:
        return state["ranked"][0] if state["ranked"] else None

    def reject_and_next(self, state: State, resname: str) -> Optional[Dict[str, Any]]:
        pname = state["project"]["name"]
        state["rejected"].setdefault(pname, []).append(resname)
        state["ranked"] = [c for c in state["ranked"] if c["resource"]["name"] != resname]
        state["pending_candidate"] = {"resource": state["ranked"][0]["resource"], "project": pname} if state["ranked"] else None
        return self.top(state)

    # -------------------- Allocation persistence (UPDATED) --------------------

    def _insert_allocation_record(self, proj: Dict[str, Any], res: Dict[str, Any], score: Optional[float] = None) -> None:
        try:
            proj_doc = self._proj_col.find_one({"name": proj.get("name")})
            res_doc  = self._res_col.find_one({"name": res.get("name")})

            if not proj_doc or not res_doc:
                print("⚠️ Allocation not saved: project or resource _id not found.")
                return

            if score is None:
                score = self._base_score(proj, res)
            fit_score = int(round((score or 0.0) * 100))

            req = set(self._norm(proj.get("required_skills", [])))
            hav = set(self._norm(res.get("skills", [])))
            matched = sorted(req & hav)
            reason = f"Matched skills: {', '.join(matched)}" if matched else "Matched on availability/location"

            alloc_doc = {
                "project_id": proj_doc["_id"],
                "resource_id": res_doc["_id"],
                "fit_score": fit_score,
                "reason": reason,
                "allocated_on": datetime.utcnow(),
                "__v": 0,
            }
            self._alloc_col.insert_one(alloc_doc)

            resource_updates = {
                "$set": {
                    "current_project": proj_doc.get("name"),
                    "is_allocated": True,
                },
                "$push": {
                    "allocations": {
                        "project_name": proj_doc.get("name"),
                        "project_id": proj_doc["_id"],
                        "fit_score": fit_score,
                        "on": datetime.utcnow(),
                        "reason": reason,
                    }
                }
            }

            if isinstance(res_doc.get("capacity_hours"), (int, float)) and res_doc["capacity_hours"] > 0:
                resource_updates.setdefault("$inc", {})["capacity_hours"] = -res_doc["capacity_hours"]

            self._res_col.update_one({"_id": res_doc["_id"]}, resource_updates)

            assigned_entry = {
                "name": res_doc.get("name"),
                "role": res_doc.get("role"),
                "skills": res_doc.get("skills"),
                "rate_per_hour": res_doc.get("rate_per_hour"),
                "location": res_doc.get("location"),
                "proficiency": res_doc.get("proficiency"),
                "allocated_on": datetime.utcnow(),
                "fit_score": fit_score,
            }
            self._proj_col.update_one(
                {"_id": proj_doc["_id"]},
                {
                    "$push": {"assigned_resources": assigned_entry},
                    "$inc": {"assigned_count": 1}
                }
            )

        except Exception as e:
            print(f"⚠️ Failed to save allocation and/or updates: {e}")

    # -------- New: Idempotent sync for confirmed assignments --------

    def _upsert_allocation_by_names(self, project_name: str, resource_name: str) -> Optional[dict]:
        try:
            proj_doc = self._proj_col.find_one({"name": project_name})
            res_doc  = self._res_col.find_one({"name": resource_name})
            if not proj_doc or not res_doc:
                print(f"⚠️ Upsert skipped: project/resource not found: {project_name} / {resource_name}")
                return None

            existing = self._alloc_col.find_one({
                "project_id": proj_doc["_id"],
                "resource_id": res_doc["_id"],
            })
            if existing:
                out = self._jsonify(existing)
                out.pop("_id", None)
                return out

            proj_clean = self._clean_project(proj_doc)
            res_clean  = self._clean_resource(res_doc)
            score = self._base_score(proj_clean, res_clean)
            fit_score = int(round((score or 0.0) * 100))

            req = set(self._norm(proj_clean.get("required_skills", [])))
            hav = set(self._norm(res_clean.get("skills", [])))
            matched = sorted(req & hav)
            reason = f"Matched skills: {', '.join(matched)}" if matched else "Matched on availability/location"

            alloc = {
                "project_id": proj_doc["_id"],
                "resource_id": res_doc["_id"],
                "fit_score": fit_score,
                "reason": reason,
                "allocated_on": datetime.utcnow(),
                "__v": 0,
            }
            self._alloc_col.insert_one(alloc)
            return self._jsonify(alloc)
        except Exception as e:
            print(f"⚠️ Upsert allocation failed: {e}")
            return None

    def sync_confirmed_assignments_to_db(self, state: "LLMService.State") -> List[dict]:
        results = []
        seen_pairs = set()
        for entry in state.get("confirmed", []):
            pname = (entry.get("project") or "").strip()
            rname = (entry.get("resource") or "").strip()
            if not pname or not rname:
                continue
            key = (pname.lower(), rname.lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            alloc = self._upsert_allocation_by_names(pname, rname)
            if alloc:
                results.append(alloc)
        return results

    # -------------------- LLM answer --------------------

    def llm_answer(
        self,
        data: Dict[str, Any],
        history_msgs: List[BaseMessage],
        query: str,
        active_project: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        top_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not LLM_OK:
            return "Install `langchain-core` + `langchain-groq` and set GROQ_API_KEY for detailed explanations."
        if not os.getenv("GROQ_API_KEY"):
            return "Please set GROQ_API_KEY in your .env file."

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a precise, context-aware project resource allocation copilot.\n"
             "Rules:\n"
             "1) Infer active project from history if user omits it; only ask if truly unknown (one short sentence).\n"
             "2) Treat required_skills as HARD for 'lowest/cheapest rate', 'highest rate', or 'best qualified' asks. "
             "If none qualify, say so, then show the closest with explicit gaps.\n"
             "3) Apply remembered constraints (location/proficiency/max_rate/skills). If conflicts arise, explain the best feasible.\n"
             "4) Do NOT select/confirm candidates; the app handles selection. You only explain/compare/clarify.\n"
             "5) Be concise; never invent data."),
            ("human",
             "Conversation (latest last):\n{history}\n\n"
             "Active project (if known): {active_project}\n"
             "Constraints for this project: {constraints}\n"
             "Top candidates (read-only):\n{top_candidates}\n"
             "Data snapshot:\n{data}\n\n"
             "User message: {query}\n\n"
             "Provide a tailored explanation.")
        ])

        hist = "\n".join(
            ("User: " + m.content) if isinstance(m, HumanMessage) else ("Assistant: " + m.content)
            for m in history_msgs[-12:]
        )

        ranked_view = json.dumps(top_candidates or [], indent=2, ensure_ascii=False, default=self._json_default)
        llm = ChatGroq(model=self.llm_model, temperature=0)
        chain = prompt | llm
        resp = chain.invoke({
            "history": hist,
            "active_project": active_project or "",
            "constraints": json.dumps(constraints or {}, indent=2, ensure_ascii=False, default=self._json_default),
            "top_candidates": ranked_view,
            "data": json.dumps(self._jsonify(data), indent=2, ensure_ascii=False, default=self._json_default),
            "query": query
        })
        return getattr(resp, "content", str(resp))

    # -------------------- Chatbot (single turn, for React/API) --------------------

    @staticmethod
    def wants_history(user_text: str) -> bool:
        t = user_text.lower()
        triggers = [
            "show history",
            "what did we discuss",
            "previous messages",
            "conversation so far",
            "what did i accept",
            "who did i accept",
            "what did we confirm",
            "summary so far",
            "recap",
        ]
        return any(phrase in t for phrase in triggers)

    @staticmethod
    def is_greeting(user_text: str) -> bool:
        t = user_text.strip().lower()
        return t in ("hi", "hello", "hey", "yo", "howdy", "sup", "good morning", "good evening", "good afternoon")

    @staticmethod
    def history_response(state: State) -> str:
        lines = []
        if state["confirmed"]:
            lines.append("Confirmed assignments:")
            for a in state["confirmed"]:
                lines.append(f"• {a['project']} → {a['resource']}")
        else:
            lines.append("No confirmed assignments yet.")
        tail = state["display_log"][-6:]
        if tail:
            lines.append("\nRecent messages:")
            for m in tail:
                who = "You" if m["role"] == "user" else "Bot"
                lines.append(f"{who}: {m['content']}")
        return "\n".join(lines)

    @staticmethod
    def describe_resource(r: Dict[str, Any]) -> str:
        return (f"Role={r.get('role')}, Skills={r.get('skills')}, Proficiency={r.get('proficiency')}, "
                f"Location={r.get('location')}, Rate=${r.get('rate_per_hour')}/hr, "
                f"Available={r.get('availability_start')}")

    @staticmethod
    def has_confirm_intent(text: str) -> bool:
        t = text.lower()
        return re.search(LLMService.CONFIRM_WORDS, t) is not None or t.strip() in ("yes", "y", "ok", "okay", "accept")

    def confirm_pending(self, state: State, proj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pname = proj["name"]
        pick_res = None
        pick_score: Optional[float] = None
        if state["pending_candidate"] and state["pending_candidate"]["project"] == pname:
            pick_res = state["pending_candidate"]["resource"]
            if state["ranked"] and state["ranked"][0]["resource"]["name"] == pick_res["name"]:
                pick_score = state["ranked"][0].get("score")
        elif state["ranked"]:
            pick_res = state["ranked"][0]["resource"]
            pick_score = state["ranked"][0].get("score")
        if not pick_res:
            return None

        state["confirmed"].append({"project": pname, "resource": pick_res["name"]})
        self._insert_allocation_record(proj, pick_res, pick_score)

        state["project"] = None
        state["ranked"] = []
        state["awaiting"] = False
        state["pending_candidate"] = None
        return pick_res

    def respond(self, user: str) -> str:
        state = self.state
        outputs: List[str] = []

        def say(text: str):
            outputs.append(text)
            state["chat_history"].append(AIMessage(content=text))
            state["display_log"].append({"role": "assistant", "content": text})

        def remember_user(text: str):
            state["chat_history"].append(HumanMessage(content=text))
            state["display_log"].append({"role": "user", "content": text})

        def set_project(proj: Dict[str, Any]):
            state["project"] = proj
            state["last_project_name"] = proj["name"]

        def ensure_active_project() -> Optional[Dict[str, Any]]:
            if state["project"]:
                return state["project"]
            if state["last_project_name"]:
                proj = self._find_project(state["projects"], state["last_project_name"])
                if proj:
                    state["project"] = proj
                    return proj
            return None

        if user.strip().lower() in ("reload db", "reload", "refresh db"):
            resources, projects = self.load_data_from_mongo()
            state["resources"] = resources
            state["projects"] = projects
            state["ranked"] = []
            state["pending_candidate"] = None
            say(f"Reloaded {len(resources)} resources & {len(projects)} projects.")
            return "\n".join(outputs)

        remember_user(user)

        if not state["resources"] or not state["projects"]:
            say("I don't see any resources/projects in Mongo yet. Please insert data and say `reload db`.")
            return "\n".join(outputs)

        inline_proj = self.detect_project_from_text(state["projects"], user)
        if inline_proj and (not state["project"] or state["project"]["name"] != inline_proj["name"]):
            set_project(inline_proj)

        if self.has_confirm_intent(user):
            proj = self.detect_project_from_text(state["projects"], user) or ensure_active_project()
            if not proj and state["pending_candidate"]:
                proj_name = state["pending_candidate"]["project"]
                proj = self._find_project(state["projects"], proj_name)
                if proj:
                    set_project(proj)
            if proj:
                confirmed = self.confirm_pending(state, proj)
                if confirmed:
                    say(f"✅ Confirmed: {proj['name']} → {confirmed['name']}\n{self.describe_resource(confirmed)}")
                    say("Tell me another project when you're ready.")
                else:
                    say("I don't have a candidate pending for confirmation. Ask me for a suggestion first.")
                return "\n".join(outputs)

        if self.is_greeting(user):
            say("Hello! Share a project name and I’ll propose the best match. You can also add constraints like 'in Hyderabad' or 'lowest rate'.")
            return "\n".join(outputs)

        if user.lstrip().startswith("{") or user.lstrip().startswith("["):
            try:
                payload = json.loads(user)
                msg = self.ingest_json_payload(state, payload)
                if state["project"]:
                    cand = self.top(state)
                    if cand:
                        r = cand["resource"]
                        state["pending_candidate"] = {"resource": r, "project": state["project"]["name"]}
                        say(f"{msg}\nRefreshed suggestion for **{state['project']['name']}** → **{r['name']}** (score {cand['score']}).\n{self.describe_resource(r)}\nAccept?")
                        state["awaiting"] = True
                    else:
                        say(msg + "\nNo qualified suggestion under current constraints; try relaxing them or update data further.")
                else:
                    say(msg + "\nUpdates noted. Share a project name when ready.")
            except json.JSONDecodeError:
                say("That looks like JSON but I couldn't parse it. Please check the brackets/quotes.")
            return "\n".join(outputs)

        if self.wants_history(user):
            synced = self.sync_confirmed_assignments_to_db(state)
            if synced:
                state["display_log"].append({"role": "assistant", "content": f"Synced {len(synced)} allocation(s) to database."})
            say(self.history_response(state))
            return "\n".join(outputs)

        if re.search(rf"\b{self.CONFIRM_WORDS}\b", user.lower()):
            chosen = None
            for r in state["resources"]:
                nm = r.get("name", "")
                if nm and nm.lower() in user.lower():
                    chosen = r
                    break
            proj = self.detect_project_from_text(state["projects"], user) or self._find_project(state["projects"], user) or ensure_active_project()
            if not proj and state["pending_candidate"]:
                proj = self._find_project(state["projects"], state["pending_candidate"]["project"])
                if proj:
                    set_project(proj)
            if proj and chosen:
                req = set(self._norm(proj.get("required_skills", [])))
                res_sk = set(self._norm(chosen.get("skills", [])))
                missing = list(req - res_sk)
                state["confirmed"].append({"project": proj["name"], "resource": chosen["name"]})
                self._insert_allocation_record(proj, chosen, self._base_score(proj, chosen))
                set_project(proj)
                warn = f" (warning: missing required skills: {missing})" if missing else ""
                say(f"✅ Confirmed by request: {proj['name']} → {chosen['name']}{warn}\n{self.describe_resource(chosen)}")
                state["project"] = None
                state["ranked"] = []
                state["awaiting"] = False
                state["pending_candidate"] = None
                say("Tell me another project when you're ready.")
                return "\n".join(outputs)
            if proj and (state["pending_candidate"] and state["pending_candidate"]["project"] == proj["name"]):
                confirmed = self.confirm_pending(state, proj)
                if confirmed:
                    say(f"✅ Confirmed: {proj['name']} → {confirmed['name']}\n{self.describe_resource(confirmed)}")
                    say("Tell me another project when you're ready.")
                else:
                    say("I don't have a candidate pending for confirmation. Ask me for a suggestion first.")
                return "\n".join(outputs)

        inferred_prefs = self.parse_preferences(user)
        if any(inferred_prefs.get(k) for k in ["hard", "soft", "skills", "mode"]):
            proj = self.detect_project_from_text(state["projects"], user) or self._find_project(state["projects"], user) or ensure_active_project()
            if not proj:
                for p in state["projects"]:
                    if p.get("name", "").lower() in user.lower():
                        proj = p
                        break
            if proj:
                pname = proj["name"]
                current = state["preferences_by_project"].get(pname, {"hard": {}, "soft": {}, "skills": []})
                current["hard"].update(inferred_prefs.get("hard", {}))
                current["soft"].update(inferred_prefs.get("soft", {}))
                cur_sk = set(current.get("skills", []))
                for s in inferred_prefs.get("skills", []):
                    cur_sk.add(s)
                current["skills"] = list(cur_sk)
                if inferred_prefs.get("mode"):
                    current["mode"] = inferred_prefs["mode"]
                state["preferences_by_project"][pname] = current
                set_project(proj)
                rejected = state["rejected"].get(pname, [])
                state["ranked"] = self.rank_for_project(proj, rejected, current, state)
                cand = self.top(state)
                if cand:
                    r = cand["resource"]
                    state["pending_candidate"] = {"resource": r, "project": pname}
                    say(f"Updated suggestion for **{pname}** → **{r['name']}** (score {cand['score']}).\n{self.describe_resource(r)}\nAccept?")
                    state["awaiting"] = True
                else:
                    say("No candidates satisfy the new constraints. Relax a requirement or try another project?")
                    state["awaiting"] = False
                    state["project"] = None
            else:
                say("Got it—tell me the project name so I can apply those constraints.")
            return "\n".join(outputs)

        if state["awaiting"] and state["project"]:
            pname = state["project"]["name"]
            if user.lower() in ("yes", "y", "ok", "okay", "accept", "accept him", "accept her", "accept it", "accept them"):
                confirmed = self.confirm_pending(state, state["project"])
                if confirmed:
                    say(f"✅ Confirmed: {pname} → {confirmed['name']}\n{self.describe_resource(confirmed)}")
                    say("Tell me another project when you're ready.")
                else:
                    say("I don't have a candidate pending for confirmation.")
                return "\n".join(outputs)

            if user.lower() in ("no", "n", "reject", "next", "try another", "someone else"):
                current = self.top(state)
                nxt = self.reject_and_next(state, current["resource"]["name"]) if current else None
                if nxt:
                    r = nxt["resource"]
                    state["pending_candidate"] = {"resource": r, "project": pname}
                    say(f"How about **{r['name']}** for {pname}? (score {nxt['score']})\n{self.describe_resource(r)}\nAccept?")
                else:
                    say("No more promising matches under current constraints. Try another project or adjust requirements.")
                    state["project"] = None
                    state["awaiting"] = False
                    state["pending_candidate"] = None
                return "\n".join(outputs)

            active_constraints = state["preferences_by_project"].get(pname, {"hard": {}, "soft": {}, "skills": []})
            ranked_view = [
                {
                    "name": c["resource"].get("name"),
                    "rate_per_hour": c["resource"].get("rate_per_hour"),
                    "location": c["resource"].get("location"),
                    "proficiency": c["resource"].get("proficiency"),
                    "skills": c["resource"].get("skills"),
                    "score": c.get("score"),
                } for c in state.get("ranked", [])[:5]
            ] if state.get("ranked") else None
            data = {"projects": state["projects"], "resources": state["resources"], "confirmed": state["confirmed"]}
            expl = self.llm_answer(self._jsonify(data), state["chat_history"], user, pname, active_constraints, ranked_view)
            say(expl)
            return "\n".join(outputs)

        if re.search(r"(who.*accepted|who.*did i accept|give me the person.*accepted)", user.lower()):
            proj = self.detect_project_from_text(state["projects"], user) or self._find_project(state["projects"], user) or ensure_active_project()
            if proj:
                accepted = [a for a in state["confirmed"] if a["project"].lower() == proj["name"].lower()]
                if accepted:
                    names = ", ".join(a["resource"] for a in accepted)
                    say(f"You have accepted for **{proj['name']}**: {names}.")
                else:
                    say(f"No one accepted yet for **{proj['name']}**.")
                return "\n".join(outputs)

        proj = self.detect_project_from_text(state["projects"], user) or self._find_project(state["projects"], user)
        if proj:
            if proj.get("status", "").lower() == "completed":
                say("That project is marked Completed. Choose another one?")
                return "\n".join(outputs)
            set_project(proj)
            prefs = state["preferences_by_project"].get(proj["name"], {"hard": {}, "soft": {}, "skills": []})
            rejected = state["rejected"].get(proj["name"], [])
            state["ranked"] = self.rank_for_project(proj, rejected, prefs, state)
            if not state["ranked"]:
                say("I don’t see strong matches yet. Add constraints (e.g., 'in Bangalore', 'lowest rate', 'max rate under 30') or load more resources.")
                state["awaiting"] = True
            else:
                cand = self.top(state)
                r = cand["resource"]
                state["pending_candidate"] = {"resource": r, "project": proj["name"]}
                say(f"Top match for **{proj['name']}** → **{r['name']}** (score {cand['score']}).\n{self.describe_resource(r)}\nAccept?")
                state["awaiting"] = True
            return "\n".join(outputs)

        active_name = state["project"]["name"] if state.get("project") else state["last_project_name"]
        active_constraints = state["preferences_by_project"].get(active_name, {"hard": {}, "soft": {}, "skills": []}) if active_name else None
        ranked_view = [
            {
                "name": c["resource"].get("name"),
                "rate_per_hour": c["resource"].get("rate_per_hour"),
                "location": c["resource"].get("location"),
                "proficiency": c["resource"].get("proficiency"),
                "skills": c["resource"].get("skills"),
                "score": c.get("score"),
            } for c in state.get("ranked", [])[:5]
        ] if state.get("ranked") else None
        data = {"projects": state["projects"], "resources": state["resources"], "confirmed": state["confirmed"]}
        expl = self.llm_answer(self._jsonify(data), state["chat_history"], user, active_name, active_constraints, ranked_view)
        say(expl)
        return "\n".join(outputs)

    # -------------------- Small helpers for your app --------------------

    def ask(self, text: str) -> dict:
        reply = self.respond(text)
        projects = [self._coerce_doc_for_json(p) for p in self._proj_col.find({}, {"_id": 0})]
        resources = [self._coerce_doc_for_json(r) for r in self._res_col.find({}, {"_id": 0})]
        allocation = [self._coerce_doc_for_json(a) for a in self._alloc_col.find({}, {"_id": 0})]
        return {
            "text": reply,
            "projects": projects,
            "resources": resources,
            "allocation": allocation,
        }

    def reset(self):
        self.state = self.new_state()

    def get_chat_history(self) -> List[Dict[str, str]]:
        return self.state["display_log"]
