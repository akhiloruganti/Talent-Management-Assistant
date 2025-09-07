import React, { useEffect, useMemo, useState } from "react";
import Navbar from "../components/Navbar";
import API from "../api/axios";

/**
 * NOTE: Core logic (data fetching, upload, auto-allocate) is unchanged.
 * This version only enhances the UI/UX using Bootstrap utilities/components.
 * To enable icons, install & import Bootstrap Icons at app entry:
 *   npm i bootstrap bootstrap-icons
 *   import 'bootstrap/dist/css/bootstrap.min.css';
 *   import 'bootstrap/dist/js/bootstrap.bundle.min.js';
 *   import 'bootstrap-icons/font/bootstrap-icons.css';
 */
const Projects = () => {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [file, setFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");

  // UI-only state (does not change core logic)
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortKey, setSortKey] = useState("recent"); // recent | name | budget

  useEffect(() => {
    const controller = new AbortController();

    const fetchProjects = async () => {
      try {
        const res = await API.get("/projects", { signal: controller.signal });
        setProjects(res.data);
        setError("");
      } catch (err) {
        if (err.name !== "CanceledError") {
          setError("Failed to fetch projects.");
          console.error(err);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
    return () => controller.abort();
  }, []);

  // Enable tooltips for any [data-bs-toggle="tooltip"]
  useEffect(() => {
    try {
      const tooltipTriggerList = Array.from(
        document.querySelectorAll('[data-bs-toggle="tooltip"]')
      );
      tooltipTriggerList.forEach((el) => new window.bootstrap.Tooltip(el));
    } catch (_) {
      /* no-op if bootstrap not loaded yet */
    }
  }, [projects]);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
  if (!file) return alert("Please select a file first!");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await API.post("/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    setUploadMessage(`✅ Uploaded: ${res.data.filename || "success"}`);

    // 🔁 Re-fetch projects so UI reflects the new data
    const refreshed = await API.get("/projects");
    setProjects(refreshed.data);

    // (optional) clear the file input
    setFile(null);
  } catch (err) {
    console.error(err);
    setUploadMessage("❌ Failed to upload file.");
  }
};


  // Auto-allocate a resource for a project (core logic unchanged)
  const handleAutoAllocate = async (projectId) => {
    try {
      const res = await API.post(`/allocations/${projectId}/auto-allocate`);
      alert(res.data.message);

      // Update projects list to reflect newly allocated resources
      const updatedProjects = projects.map((proj) =>
        proj._id === projectId
          ? { ...proj, resources: [res.data.allocatedResource] }
          : proj
      );
      setProjects(updatedProjects);
    } catch (err) {
      console.error(err);
      alert("Error allocating resource: " + (err.response?.data?.message || err.message));
    }
  };

  // ---- UI helpers (formatters) ----
  const formatDate = (iso) =>
    iso ? new Date(iso).toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" }) : "N/A";

  const money = (n) =>
    typeof n === "number"
      ? new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n)
      : "$0";

  const statusBadge = (statusRaw) => {
    const s = (statusRaw || "").toLowerCase();
    const map = {
      completed: "success",
      "in progress": "warning",
      "on hold": "secondary",
      cancelled: "danger",
      planned: "info",
    };
    return map[s] || "secondary";
  };

  // ---- Derived, display-only list ----
  const filteredSorted = useMemo(() => {
    let list = [...projects];

    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter(
        (p) =>
          p.name?.toLowerCase().includes(q) ||
          p.client_name?.toLowerCase().includes(q) ||
          p.description?.toLowerCase().includes(q) ||
          (p.required_skills || []).join(",").toLowerCase().includes(q)
      );
    }

    if (statusFilter !== "all") {
      list = list.filter((p) => (p.status || "").toLowerCase() === statusFilter);
    }

    if (sortKey === "name") {
      list.sort((a, b) => (a.name || "").localeCompare(b.name || ""));
    } else if (sortKey === "budget") {
      list.sort((a, b) => (b.budget || 0) - (a.budget || 0));
    } else {
      // recent: sort by start_date desc (fallback to createdAt if present)
      list.sort(
        (a, b) =>
          new Date(b.start_date || b.createdAt || 0) - new Date(a.start_date || a.createdAt || 0)
      );
    }
    return list;
  }, [projects, query, statusFilter, sortKey]);

  if (loading) {
    return (
      <div className="d-flex vh-100 bg-light">
        <div className="m-auto text-center">
          <div className="spinner-border" role="status" aria-hidden="true" />
          <div className="mt-3 fw-semibold">Loading projects…</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <>
        <Navbar />
        <div className="container py-4">
          <div className="alert alert-danger d-flex align-items-center gap-2" role="alert">
            <i className="bi bi-exclamation-triangle-fill"></i>
            <span>{error}</span>
          </div>
        </div>
      </>
    );
  }

  const uploadAlertClass = uploadMessage.startsWith("✅")
    ? "alert-success"
    : uploadMessage.startsWith("❌")
    ? "alert-danger"
    : "alert-secondary";

  return (
    <div className="bg-light min-vh-100">
      <Navbar />
      <div className="container py-4">
        {/* Page Header */}
        <div className="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3">
          <div className="d-flex align-items-center gap-3">
            <h1 className="h3 mb-0">Projects</h1>
            <span className="badge text-bg-primary">{projects.length} total</span>
          </div>

          {/* Toolbar: Search / Filter / Sort */}
          <div className="d-flex flex-wrap align-items-center gap-2">
            <div className="input-group">
              <span className="input-group-text bg-white">
                <i className="bi bi-search"></i>
              </span>
              <input
                className="form-control"
                placeholder="Search by name, client, or skill…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>

            <select
              className="form-select"
              style={{ minWidth: 160 }}
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              title="Filter by status"
            >
              <option value="all">All statuses</option>
              <option value="planned">Planned</option>
              <option value="in progress">In Progress</option>
              <option value="completed">Completed</option>
              <option value="on hold">On Hold</option>
              <option value="cancelled">Cancelled</option>
            </select>

            <select
              className="form-select"
              style={{ minWidth: 140 }}
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value)}
              title="Sort"
            >
              <option value="recent">Sort: Recent</option>
              <option value="name">Sort: Name</option>
              <option value="budget">Sort: Budget</option>
            </select>
          </div>
        </div>

        {/* Upload Card */}
        <div className="card mb-4 shadow-sm">
          <div className="card-body">
            <div className="d-flex align-items-center justify-content-between mb-3">
              <h3 className="h5 mb-0">
                <i className="bi bi-upload me-2"></i>Upload a File
              </h3>
              <span className="text-muted small">CSV / XLSX / PDF supported (as per backend)</span>
            </div>
            <div className="row g-2 align-items-center">
              <div className="col-12 col-md-8">
                <div className="input-group">
                  <label className="input-group-text" htmlFor="projectFile">
                    <i className="bi bi-paperclip"></i>
                  </label>
                  <input
                    id="projectFile"
                    type="file"
                    className="form-control"
                    onChange={handleFileChange}
                  />
                  <button
                    onClick={handleUpload}
                    disabled={!file}
                    className={`btn btn-primary ${!file ? "disabled" : ""}`}
                    type="button"
                  >
                    <i className="bi bi-cloud-arrow-up me-1"></i>
                    Upload
                  </button>
                </div>
              </div>
              <div className="col-12 col-md-4">
                {uploadMessage && (
                  <div className={`alert ${uploadAlertClass} py-2 mb-0 d-flex align-items-center gap-2`} role="alert">
                    <i className="bi bi-info-circle"></i>
                    <span>{uploadMessage}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Projects Grid */}
        <div className="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4">
          {filteredSorted.map((project) => (
            <div className="col" key={project._id}>
              <div className="card h-100 shadow-sm border-0">
                {/* Card header bar */}
                <div className="px-3 pt-3 d-flex justify-content-between align-items-start">
                  <h2 className="h5 mb-2">{project.name}</h2>
                  <span
                    className={`badge text-bg-${statusBadge(project.status)}`}
                    data-bs-toggle="tooltip"
                    title={`Status: ${project.status || "N/A"}`}
                  >
                    {project.status || "N/A"}
                  </span>
                </div>

                <div className="card-body pt-2 d-flex flex-column">
                  <p className="text-muted small mb-3">
                    {project.description || "No description provided."}
                  </p>

                  {/* Key facts */}
                  <div className="row g-2 mb-3 small">
                    <div className="col-6">
                      <div className="d-flex align-items-center gap-2">
                        <i className="bi bi-calendar-event"></i>
                        <span><strong>Start:</strong> {formatDate(project.start_date)}</span>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="d-flex align-items-center gap-2">
                        <i className="bi bi-calendar2-check"></i>
                        <span><strong>End:</strong> {formatDate(project.end_date)}</span>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="d-flex align-items-center gap-2">
                        <i className="bi bi-cash-coin"></i>
                        <span><strong>Budget:</strong> {money(project.budget || 0)}</span>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="d-flex align-items-center gap-2">
                        <i className="bi bi-person-badge"></i>
                        <span><strong>Client:</strong> {project.client_name || "N/A"}</span>
                      </div>
                    </div>
                    <div className="col-12">
                      <div className="d-flex align-items-center gap-2">
                        <i className="bi bi-geo-alt"></i>
                        <span><strong>Location:</strong> {project.location || "N/A"}</span>
                      </div>
                    </div>
                  </div>

                  {/* Skills chips */}
                  <div className="mb-3">
                    <div className="mb-1 fw-semibold small text-uppercase text-secondary">Required Skills</div>
                    {project.required_skills?.length ? (
                      <div className="d-flex flex-wrap gap-2">
                        {project.required_skills.map((sk, idx) => (
                          <span key={idx} className="badge rounded-pill text-bg-light border">
                            <i className="bi bi-tools me-1"></i>
                            {sk}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-muted small">N/A</span>
                    )}
                  </div>

                  {/* Assigned resources */}
                  <div className="mb-3">
                    <div className="mb-1 fw-semibold small text-uppercase text-secondary">Resources</div>
                    {project.resources?.length ? (
                      <div className="d-flex flex-wrap gap-2">
                        {project.resources.map((r, idx) => (
                          <span key={idx} className="badge rounded-pill text-bg-secondary">
                            <i className="bi bi-people me-1"></i>
                            {r.name}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-muted small">None yet</span>
                    )}
                  </div>

                  <div className="mt-auto">
                    <button
                      onClick={() => handleAutoAllocate(project._id)}
                      className="btn btn-success w-100"
                      type="button"
                      data-bs-toggle="tooltip"
                      title="Let the system pick the best resource for this project"
                    >
                      <i className="bi bi-magic me-2"></i>
                      Allocate Best Resource
                    </button>
                  </div>
                </div>

                {/* Subtle footer */}
                <div className="card-footer bg-body-tertiary small text-muted d-flex justify-content-between">
                  <span>
                    <i className="bi bi-hash me-1"></i>
                    {project._id?.slice?.(0, 8) || "—"}
                  </span>
                  <span>
                    <i className="bi bi-clock-history me-1"></i>
                    Updated {formatDate(project.updatedAt || project.start_date)}
                  </span>
                </div>
              </div>
            </div>
          ))}

          {filteredSorted.length === 0 && (
            <div className="col">
              <div className="alert alert-info d-flex align-items-center gap-2" role="alert">
                <i className="bi bi-info-circle-fill"></i>
                <span>No projects match your current filters.</span>
              </div>
            </div>
          )}
        </div>

        {/* Back to top */}
        <div className="d-flex justify-content-end mt-4">
          <a href="#top" className="btn btn-outline-secondary">
            <i className="bi bi-arrow-up-short me-1"></i>
            Back to top
          </a>
        </div>
      </div>
    </div>
  );
};

export default Projects;
