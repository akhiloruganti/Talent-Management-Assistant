import React, { useState, useEffect, useRef } from "react";
import Navbar from "../components/Navbar";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import axios from "axios";
import "./Chatbot.css";

const API_BASE = "http://localhost:8000";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [chatWidth, setChatWidth] = useState(300);
  const [resizing, setResizing] = useState(false);
  const resizerRef = useRef(null);
  const endRef = useRef(null);

  const [aiData, setAiData] = useState({ gantt: [], heatmap: [] });

  // NEW: allocations + latest catalogs so we can resolve names
  const [allocations, setAllocations] = useState([]);
  const [catalogProjects, setCatalogProjects] = useState([]);
  const [catalogResources, setCatalogResources] = useState([]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initial fetch: catalog (projects/resources) for name resolution & charts
  useEffect(() => {
    const fetchCatalog = async () => {
      try {
        const catalogRes = await axios.get(`${API_BASE}/catalog`);
        const { projects = [], resources = [] } = catalogRes.data || {};

        setCatalogProjects(projects);
        setCatalogResources(resources);

        // Build a simple Gantt-like series from project estimates if present
        const gantt = (projects || []).map((p) => ({
          name: p.name || "Unnamed",
          hours: Number(p.estimated_hours || 0),
        }));

        // Skills count (heatmap)
        const skillCounts = {};
        (resources || []).forEach((r) => {
          (r.skills || []).forEach((s) => {
            const key = String(s || "").trim();
            if (!key) return;
            skillCounts[key] = (skillCounts[key] || 0) + 1;
          });
        });
        const heatmapArray = Object.keys(skillCounts).map((skill) => ({
          skill,
          usage: skillCounts[skill],
        }));

        setAiData({ gantt, heatmap: heatmapArray });
      } catch (error) {
        console.error("Error fetching /catalog:", error);
      }
    };

    fetchCatalog();
  }, []);

  // Helper: id → name maps (works if backend includes _id on catalogs)
  const projectNameById = React.useMemo(() => {
    const m = new Map();
    (catalogProjects || []).forEach((p) => {
      if (p && (p._id || p.id)) m.set(String(p._id || p.id), p.name || "Unnamed");
    });
    return m;
  }, [catalogProjects]);

  const resourceNameById = React.useMemo(() => {
    const m = new Map();
    (catalogResources || []).forEach((r) => {
      if (r && (r._id || r.id)) m.set(String(r._id || r.id), r.name || "Unnamed");
    });
    return m;
  }, [catalogResources]);

  // “Denormalize” each allocation row to show friendly names
  const allocationCards = React.useMemo(() => {
    return (allocations || []).map((a, idx) => {
      const pid = String(a.project_id || "");
      const rid = String(a.resource_id || "");
      const projectName =
        a.project_name || projectNameById.get(pid) || (pid ? `Project ${pid.slice(-6)}` : "Project —");
      const resourceName =
        a.resource_name || resourceNameById.get(rid) || (rid ? `Resource ${rid.slice(-6)}` : "Resource —");
      const score = typeof a.fit_score === "number" ? a.fit_score : undefined;
      const reason = a.reason || "";
      const when = a.allocated_on ? new Date(a.allocated_on) : null;
      const whenLabel = when ? when.toLocaleString() : "";

      return {
        key: `${pid}-${rid}-${idx}`,
        projectName,
        resourceName,
        score,
        reason,
        whenLabel,
      };
    });
  }, [allocations, projectNameById, resourceNameById]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;

    setMessages((prev) => [...prev, { sender: "user", text }]);
    setInput("");

    try {
      const res = await axios.post(`${API_BASE}/chat`, { message: text });

      // Backend returns: { text, projects, resources, allocation, gantt?, heatmap? }
      const {
        text: aiText,
        projects,
        resources,
        allocation,
        gantt,
        heatmap,
      } = res.data || {};

      // Show the AI text
      setMessages((prev) => [...prev, { sender: "ai", text: aiText || "(no response)" }]);

      // Keep the latest catalogs for name resolution (if backend returns them)
      if (Array.isArray(projects) && projects.length) setCatalogProjects(projects);
      if (Array.isArray(resources) && resources.length) setCatalogResources(resources);

      // NEW: save allocations (cards use this)
      if (Array.isArray(allocation)) setAllocations(allocation);

      // Prefer backend-provided charts if present; otherwise recompute from payloads
      if (Array.isArray(gantt) || Array.isArray(heatmap)) {
        setAiData((prev) => ({
          gantt: Array.isArray(gantt) ? gantt : prev.gantt,
          heatmap: Array.isArray(heatmap) ? heatmap : prev.heatmap,
        }));
      } else if (Array.isArray(projects) || Array.isArray(resources)) {
        const gantt2 = (projects || []).map((p) => ({
          name: p.name || "Unnamed",
          hours: Number(p.estimated_hours || 0),
        }));
        const skillCounts = {};
        (resources || []).forEach((r) => {
          (r.skills || []).forEach((s) => {
            const key = String(s || "").trim();
            if (!key) return;
            skillCounts[key] = (skillCounts[key] || 0) + 1;
          });
        });
        const heatmap2 = Object.keys(skillCounts).map((skill) => ({
          skill,
          usage: skillCounts[skill],
        }));
        setAiData({ gantt: gantt2, heatmap: heatmap2 });
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, { sender: "ai", text: "Error: backend not responding" }]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleMouseDown = () => setResizing(true);
  const handleMouseMove = (e) => {
    if (resizing) {
      const newWidth = e.clientX;
      if (newWidth > 240 && newWidth < 640) setChatWidth(newWidth);
    }
  };
  const handleMouseUp = () => setResizing(false);

  return (
    <div onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} style={{ height: "100vh", overflow: "hidden" }}>
      <Navbar />
      <div className="chat-container">
        {/* Chat Pane */}
        <div className="chat-left" style={{ width: chatWidth }}>
          <h2>Chat</h2>
          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={msg.sender === "user" ? "message user" : "message ai"}>
                {msg.text}
              </div>
            ))}
            <div ref={endRef} />
          </div>
          <div className="chat-input">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
            />
            <button onClick={handleSend}>Send</button>
          </div>
        </div>

        {/* Resizer */}
        <div ref={resizerRef} className="resizer" onMouseDown={handleMouseDown}></div>

        {/* Results Pane */}
        <div className="chat-right">
          <h2>Allocations</h2>

          {/* === Allocation Cards (TOP) === */}
          <div className="allocations-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: "12px", marginBottom: "24px" }}>
            {allocationCards.length === 0 ? (
              <div style={{ opacity: 0.7 }}>No allocations yet. Confirm one in chat to see it here.</div>
            ) : (
              allocationCards.map((card) => (
                <div key={card.key} className="alloc-card" style={{ border: "1px solid #e6e6e6", borderRadius: 10, padding: 12, background: "#fff" }}>
                  <div style={{ fontWeight: 600, marginBottom: 6 }}>
                    {card.projectName} <span style={{ opacity: 0.6 }}>→</span> {card.resourceName}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 12, opacity: 0.7 }}>{card.whenLabel}</span>
                    {typeof card.score === "number" && (
                      <span style={{ fontSize: 12, padding: "2px 8px", borderRadius: 999, background: "#f1f5ff", border: "1px solid #dbe4ff" }}>
                        fit: {card.score}
                      </span>
                    )}
                  </div>
                  {card.reason && (
                    <div style={{ fontSize: 13, opacity: 0.9 }}>
                      <strong>Reason:</strong> {card.reason}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          {/* === Charts (UNDER the cards) === */}
          <div style={{ marginBottom: "2rem" }}>
            <h4>Gantt Chart (Project Hours)</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={aiData.gantt}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="hours" fill="#007bff" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div>
            <h4>Skill Usage Heatmap</h4>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={aiData.heatmap}>
                <XAxis dataKey="skill" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="usage" fill="#28a745" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
