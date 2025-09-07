import React, { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import API from "../api/axios";

const Resources = () => {
  const [resources, setResources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Upload states (core logic unchanged)
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");

  useEffect(() => {
    const fetchResources = async () => {
      try {
        const res = await API.get("/resources");
        setResources(res.data);
      } catch (err) {
        setError("Failed to fetch resources.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchResources();
  }, []);

  const handleUpload = async () => {
    if (!file) {
      setUploadError("Please select a file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      setUploadError("");
      setUploadMessage("");

      const res = await API.post("/upload/resource", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setUploadMessage(`✅ Uploaded: ${res.data.filename || "success"}`);
      // Append new resources (core logic unchanged)
      setResources((prev) => [...prev, ...(res.data.resources || [])]);
      setFile(null);
    } catch (err) {
      console.error(err);
      setUploadError("Upload failed. Please check the file format or try again.");
    } finally {
      setUploading(false);
    }
  };

  // Loading screen
  if (loading) {
    return (
      <div className="bg-light min-vh-100">
        <Navbar />
        <div className="container py-5">
          <div className="d-flex justify-content-center align-items-center py-5">
            <div className="text-center">
              <div className="spinner-border" role="status" aria-hidden="true" />
              <div className="mt-3 fw-semibold">Loading resources…</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const messageClass =
    uploadMessage.startsWith("✅") ? "alert-success" :
    uploadMessage.startsWith("❌") ? "alert-danger" :
    "alert-secondary";

  return (
    <div className="bg-light min-vh-100">
      <Navbar />
      <div className="container py-4">
        {/* Page Header */}
        <div className="d-flex align-items-center justify-content-between mb-3">
          <h1 className="h3 mb-0">Resources</h1>
          <span className="badge text-bg-primary">
            {resources.length} total
          </span>
        </div>

        {/* Upload Section */}
        <div className="card shadow-sm mb-4">
          <div className="card-body">
            <h2 className="h5 mb-3">Upload Resource File</h2>
            <div className="row g-2 align-items-center">
              <div className="col-12 col-md-8">
                <div className="input-group">
                  <input
                    type="file"
                    className="form-control"
                    onChange={(e) => setFile(e.target.files[0])}
                    accept=".csv,.xlsx,.xls,.pdf"
                  />
                  <button
                    type="button"
                    onClick={handleUpload}
                    className="btn btn-primary"
                    disabled={uploading || !file}
                  >
                    {uploading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true" />
                        Uploading…
                      </>
                    ) : (
                      "Upload"
                    )}
                  </button>
                </div>
              </div>
              <div className="col-12 col-md-4">
                {uploadError && (
                  <div className="alert alert-danger py-2 mb-0" role="alert">
                    {uploadError}
                  </div>
                )}
                {!uploadError && uploadMessage && (
                  <div className={`alert ${messageClass} py-2 mb-0`} role="alert">
                    {uploadMessage}
                  </div>
                )}
              </div>
            </div>
            <div className="form-text mt-2">
              Supported: .csv, .xlsx, .xls, .pdf
            </div>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        )}

        {/* Resource Cards */}
        {resources.length === 0 ? (
          <div className="alert alert-info" role="alert">
            No resources found.
          </div>
        ) : (
          <div className="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4">
            {resources.map((res) => (
              <div className="col" key={res._id}>
                <div className="card h-100 shadow-sm">
                  <div className="card-body d-flex flex-column">
                    <div className="d-flex justify-content-between align-items-start mb-2">
                      <h3 className="h5 mb-0">{res.name}</h3>
                      <span className="badge text-bg-secondary">{res.role || "—"}</span>
                    </div>

                    <ul className="list-group list-group-flush mb-3">
                      <li className="list-group-item px-0">
                        <strong>Skills:</strong>{" "}
                        {res.skills?.length ? res.skills.join(", ") : "N/A"}
                      </li>
                      <li className="list-group-item px-0">
                        <strong>Proficiency:</strong> {res.proficiency || "N/A"}
                      </li>
                      <li className="list-group-item px-0">
                        <strong>Availability:</strong>{" "}
                        {res.availability_start
                          ? new Date(res.availability_start).toLocaleDateString()
                          : "N/A"}
                      </li>
                      <li className="list-group-item px-0">
                        <strong>Location:</strong> {res.location || "N/A"}
                      </li>
                      <li className="list-group-item px-0">
                        <strong>Rate per Hour:</strong>{" "}
                        {typeof res.rate_per_hour === "number" ? `$${res.rate_per_hour}` : "N/A"}
                      </li>
                      <li className="list-group-item px-0">
                        <strong>Capacity Hours:</strong>{" "}
                        {res.capacity_hours ?? "N/A"}
                      </li>
                    </ul>

                    {/* spacer to keep equal heights if needed */}
                    <div className="mt-auto d-flex justify-content-end">
                      {/* Placeholder for future actions */}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Resources;
