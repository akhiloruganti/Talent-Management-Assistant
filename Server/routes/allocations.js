import express from "express";
import Allocation from "../models/Allocation.js";
import Project from "../models/Project.js";
import Resource from "../models/Resource.js";

const router = express.Router();

// 1️⃣ Preview allocation
router.get("/:projectId/preview", async (req, res) => {
  try {
    const { projectId } = req.params;
    const project = await Project.findById(projectId);
    if (!project) return res.status(404).json({ message: "Project not found" });

    const resources = await Resource.find();

    // Simple fit-score logic
    let bestResource = null;
    let maxScore = -1;
    let reason = "";

    resources.forEach((resr) => {
      const matchSkills = resr.skills.filter((s) =>
        project.required_skills.includes(s)
      );
      const score = matchSkills.length; // simple fit-score
      if (score > maxScore) {
        maxScore = score;
        bestResource = resr;
        reason = `Matched skills: ${matchSkills.join(", ")}`;
      }
    });

    if (!bestResource)
      return res.status(404).json({ message: "No suitable resource found" });

    res.json({
      project,
      bestResource,
      fit_score: maxScore,
      reason,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Server error" });
  }
});

// 2️⃣ Confirm allocation
router.post("/:projectId/confirm", async (req, res) => {
  try {
    const { projectId } = req.params;
    const { resourceId, fit_score, reason } = req.body;

    const allocation = new Allocation({
      project_id: projectId,
      resource_id: resourceId,
      fit_score,
      reason,
    });

    await allocation.save();

    // Optional: update project.resources array
    await Project.findByIdAndUpdate(projectId, {
      $push: { resources: resourceId },
    });

    res.json({ message: "Allocation confirmed", allocation });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Server error" });
  }
});

export default router;
