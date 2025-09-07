import Resource from "../models/Resource.js";
import Project from "../models/Project.js";
import Allocation from "../models/Allocation.js";

// Calculate fit score (example: skills + proficiency + location)
const calculateFitScore = (resource, project) => {
  let score = 0;

  // Skills match (each matched skill adds points)
  const matchedSkills = project.required_skills.filter(skill =>
    resource.skills.includes(skill)
  );
  score += matchedSkills.length * 10;

  // Proficiency weight
  const proficiencyWeights = { Intern: 1, Mid: 2, Senior: 3 };
  score += proficiencyWeights[resource.proficiency] * 5;

  // Location match bonus
  if (resource.location === project.location) score += 5;

  return score;
};

export const autoAllocate = async (projectId) => {
  // 1. Load project
  const project = await Project.findById(projectId);
  if (!project) throw new Error("Project not found");

  // 2. Get available resources
  const candidates = await Resource.find({
    availability_start: { $lte: project.start_date },
  });

  if (!candidates.length) throw new Error("No resources available");

  // 3. Calculate fit scores
  let bestMatch = null;
  let highestScore = -1;

  for (let resource of candidates) {
    const score = calculateFitScore(resource, project);

    if (score > highestScore) {
      highestScore = score;
      bestMatch = resource;
    }
  }

  if (!bestMatch) throw new Error("No suitable resource found");

  // 4. Save allocation
  const allocation = new Allocation({
    project_id: project._id,
    resource_id: bestMatch._id,
    fit_score: highestScore,
    reason: `Matched skills: ${project.required_skills.filter(skill =>
      bestMatch.skills.includes(skill)
    ).join(", ")}`
  });

  await allocation.save();

  return { allocation, project, bestResource: bestMatch };
};
