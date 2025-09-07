import express from "express";
import Resource from "../models/Resource.js";

const router = express.Router();

router.get("/", async (req, res) => {
  const resources = await Resource.find();
  res.json(resources);
});

router.post("/", async (req, res) => {
  const resource = new Resource(req.body);
  await resource.save();
  res.json({ message: "Resource added", resource });
});

export default router;
