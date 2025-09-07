import mongoose from "mongoose";

const ProjectSchema = new mongoose.Schema({
  name: { type: String, required: true },
  description: { type: String },
  required_skills: [{ type: String }],
  start_date: { type: Date, required: true },
  end_date: { type: Date },
  budget: { type: Number, default: 0 },
  resources: [{ type: mongoose.Schema.Types.ObjectId, ref: "Resource" }],
  status: { 
    type: String, 
    enum: ["Not Started", "In Progress", "Completed", "On Hold"], 
    default: "Not Started" 
  },
  client_name: { type: String },
  location: { type: String }
}, { timestamps: true });

const Project = mongoose.model("Project", ProjectSchema);
export default Project;
