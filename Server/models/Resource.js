import mongoose from "mongoose";

const ResourceSchema = new mongoose.Schema({
  name: { type: String, required: true },
  role: { type: String, required: true },
  skills: [{ type: String }],
  proficiency: { type: String, enum: ["Intern", "Mid", "Senior"], required: true },
  capacity_hours: { type: Number, default: 40 },
  availability_start: { type: Date, required: true },
  location: { type: String },
  rate_per_hour: { type: Number, default: 0 },
  current_project: { type: mongoose.Schema.Types.ObjectId, ref: "Project", default: null }
});

const Resource = mongoose.model("Resource", ResourceSchema);
export default Resource;
