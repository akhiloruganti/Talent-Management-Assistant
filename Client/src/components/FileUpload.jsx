import React, { useState } from "react";
import API from "../api/axios";
import "./FileUpload.css";

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setStatus("");
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setStatus("Uploading...");
      const response = await API.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setStatus(`Upload successful: ${response.data.message}`);
      setFile(null);
    } catch (error) {
      console.error(error);
      setStatus("Upload failed. Try again.");
    }
  };

  return (
    <div className="file-upload">
      <h3>Upload Resource / Project File</h3>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      {status && <p className="status">{status}</p>}
    </div>
  );
};

export default FileUpload;
