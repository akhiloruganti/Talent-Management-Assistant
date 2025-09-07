import express from "express";
import Project from "../models/Project.js";

const router = express.Router();

router.get("/", async (req, res) => {
  const projects = await Project.find();
  res.json(projects);
});

router.post("/", async (req, res) => {
  const project = new Project(req.body);
  await project.save();
  res.json({ message: "Project added", project });
});

export default router;
