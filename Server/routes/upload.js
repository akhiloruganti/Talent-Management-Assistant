import express from "express";
import multer from "multer";
import fs from "fs";
import axios from "axios";
import FormData from "form-data";

const router = express.Router();
const upload = multer({ dest: "uploads/" });

// ------------------- Project Upload -------------------
router.post("/", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ message: "No file uploaded" });

  try {
    const form = new FormData();
    form.append("file", fs.createReadStream(req.file.path), req.file.originalname);

    const pythonUrl = "http://localhost:8000/uploadprojectfile?persist=true";
    const response = await axios.post(pythonUrl, form, { headers: form.getHeaders() });

    fs.unlinkSync(req.file.path);

    // Send Python response directly to React (keeps res.data.projects intact)
    res.json(response.data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Failed to forward file to Python backend", error: err.message });
  }
});

// ------------------- Resource Upload -------------------
router.post("/resource", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ message: "No file uploaded" });

  try {
    const form = new FormData();
    form.append("file", fs.createReadStream(req.file.path), req.file.originalname);

    const pythonUrl = "http://localhost:8000/upload_resource_file?persist=true";
    const response = await axios.post(pythonUrl, form, { headers: form.getHeaders() });

    fs.unlinkSync(req.file.path);

    // Send Python response directly to React (keeps res.data.resources intact)
    res.json(response.data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Failed to forward file to Python backend", error: err.message });
  }
});

export default router;
