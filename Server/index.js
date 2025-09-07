import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import mongoose from "mongoose";

// Models
import Project from "./models/Project.js";
import Resource from "./models/Resource.js";

// Routes
import resourceRoutes from "./routes/resources.js";
import projectRoutes from "./routes/projects.js";
import allocationRoutes from "./routes/allocations.js";
import uploadRoutes from "./routes/upload.js";
import allocateRoutes from "./routes/allocate.js";



dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Fix __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use("/upload", uploadRoutes);
// Set EJS as template engine (for temporary rendering)
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// Static files
app.use(express.static(path.join(__dirname, "public")));

// MongoDB connection
const connectDB = async () => {
  try {
    await mongoose.connect("mongodb://localhost:27017/hackathon");
    console.log("MongoDB connected");
  } catch (err) {
    console.error("MongoDB connection failed:", err);
    process.exit(1);
  }
};

// ------------------ API ROUTES ------------------
app.use("/resources", resourceRoutes);
app.use("/projects", projectRoutes);
app.use("/allocations", allocationRoutes);
app.use("/upload", uploadRoutes);
app.use("/allocate", allocateRoutes);

// ------------------ TEMPORARY RENDER ROUTES ------------------

// Projects table (EJS view)
app.get("/projects/view", async (req, res) => {
  try {
    const projects = await Project.find();
    res.render("projects", { projects });
  } catch (err) {
    console.error(err);
    res.status(500).send("Failed to fetch projects");
  }
});

// Resources table (EJS view)
app.get("/resources/view", async (req, res) => {
  try {
    const resources = await Resource.find();
    res.render("resources", { resources });
  } catch (err) {
    console.error(err);
    res.status(500).send("Failed to fetch resources");
  }
});

// ------------------ START SERVER ------------------
connectDB().then(() => {
  app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
});
