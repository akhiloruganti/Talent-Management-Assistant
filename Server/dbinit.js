import mongoose from "mongoose";
import dotenv from "dotenv";

import Project from "./models/Project.js";
import Resource from "./models/Resource.js";
import Allocation from "./models/Allocation.js";

dotenv.config();

const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI || "mongodb://localhost:27017/hackathon");
    console.log("MongoDB connected");
  } catch (err) {
    console.error("MongoDB connection failed:", err);
    process.exit(1);
  }
};

const seedDB = async () => {
  try {
    await connectDB();

    // Clear existing collections
    await Project.deleteMany();
    await Resource.deleteMany();
    await Allocation.deleteMany();

    // Insert dummy projects
    // const projects = await Project.insertMany([
    //   {
    //     name: "AI Computer Vision",
    //     description: "Project to develop CV models for object detection",
    //     required_skills: ["AI", "Computer Vision", "Python"],
    //     start_date: new Date("2025-09-10"),
    //     end_date: new Date("2025-12-15"),
    //     budget: 50000,
    //     resources: [],
    //     status: "Not Started",
    //     client_name: "Client A",
    //     location: "Remote"
    //   },
    //   {
    //     name: "Full-Stack Web App",
    //     description: "Develop a full-stack web app for internal tools",
    //     required_skills: ["React", "Node.js", "MongoDB"],
    //     start_date: new Date("2025-09-15"),
    //     end_date: new Date("2025-11-30"),
    //     budget: 40000,
    //     resources: [],
    //     status: "Not Started",
    //     client_name: "Client B",
    //     location: "US"
    //   },
    //   {
    //     name: "Data Analytics Dashboard",
    //     description: "Create an analytics dashboard for business insights",
    //     required_skills: ["Python", "Pandas", "Visualization"],
    //     start_date: new Date("2025-10-01"),
    //     end_date: new Date("2025-12-31"),
    //     budget: 30000,
    //     resources: [],
    //     status: "Not Started",
    //     client_name: "Client C",
    //     location: "India"
    //   }
    // ]);

    // // Insert dummy resources
    // const resources = await Resource.insertMany([
    //   {
    //     name: "Alice Johnson",
    //     role: "AI Engineer",
    //     skills: ["AI", "Computer Vision"],
    //     proficiency: "Senior",
    //     capacity_hours: 40,
    //     availability_start: new Date("2025-09-10"),
    //     location: "Remote",
    //     rate_per_hour: 50,
    //     current_project: null
    //   },
    //   {
    //     name: "Bob Smith",
    //     role: "Full-Stack Developer",
    //     skills: ["React", "Node.js", "MongoDB"],
    //     proficiency: "Mid",
    //     capacity_hours: 35,
    //     availability_start: new Date("2025-09-12"),
    //     location: "US",
    //     rate_per_hour: 40,
    //     current_project: null
    //   },
    //   {
    //     name: "Charlie Davis",
    //     role: "Data Analyst",
    //     skills: ["Python", "Pandas", "Visualization"],
    //     proficiency: "Intern",
    //     capacity_hours: 20,
    //     availability_start: new Date("2025-09-20"),
    //     location: "India",
    //     rate_per_hour: 20,
    //     current_project: null
    //   }
    // ]);

    // // Insert dummy allocations
    // await Allocation.insertMany([
    //   {
    //     project_id: projects[0]._id, // AI Computer Vision
    //     resource_id: resources[0]._id, // Alice
    //     fit_score: 95,
    //     reason: "Skill match and availability"
    //   },
    //   {
    //     project_id: projects[1]._id, // Full-Stack Web App
    //     resource_id: resources[1]._id, // Bob
    //     fit_score: 90,
    //     reason: "Skill match and capacity"
    //   },
    //   {
    //     project_id: projects[2]._id, // Data Analytics Dashboard
    //     resource_id: resources[2]._id, // Charlie
    //     fit_score: 85,
    //     reason: "Skill match and availability"
    //   }
    // ]);

    console.log("Dummy data inserted successfully!");
    process.exit(0);
  } catch (err) {
    console.error(err);
    process.exit(1);
  }
};

seedDB();
