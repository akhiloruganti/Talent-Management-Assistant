import express from "express";
import axios from "axios";

const router = express.Router();

router.post("/", async (req, res) => {
  try {
    const pythonAIUrl = "http://localhost:8001/allocate"; // Python AI endpoint
    const response = await axios.post(pythonAIUrl);
    res.json(response.data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Failed to get AI allocations" });
  }
});

export default router;
