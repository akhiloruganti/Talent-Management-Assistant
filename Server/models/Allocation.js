import mongoose from "mongoose";

const AllocationSchema = new mongoose.Schema({
  project_id: { type: mongoose.Schema.Types.ObjectId, ref: "Project", required: true },
  resource_id: { type: mongoose.Schema.Types.ObjectId, ref: "Resource", required: true },
  fit_score: { type: Number, required: true },
  reason: { type: String },
  allocated_on: { type: Date, default: Date.now }
});

const Allocation = mongoose.model("Allocation", AllocationSchema);
export default Allocation;
