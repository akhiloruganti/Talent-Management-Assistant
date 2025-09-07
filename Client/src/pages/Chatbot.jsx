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

  const [aiData, setAiData] = useState({
    gantt: [],
    heatmap: [],
  });

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initial fetch: catalog only (your backend doesn’t expose GET /assignments)
  useEffect(() => {
    const fetchCatalog = async () => {
      try {
        const catalogRes = await axios.get(`${API_BASE}/catalog`);
        const { projects = [], resources = [] } = catalogRes.data || {};

        // Build a simple Gantt-like data from project estimates if present
        const gantt = (projects || []).map((p) => ({
          name: p.name || "Unnamed",
          hours: Number(p.estimated_hours || 0),
        }));

        // If you want a skills heatmap without /assignments, estimate from resources
        // (counts how many people list each skill)
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

  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;

    setMessages((prev) => [...prev, { sender: "user", text }]);
    setInput("");

    try {
      const res = await axios.post(`${API_BASE}/chat`, { message: text });
      // Backend returns: { text, projects, resources, allocation, gantt?, heatmap? }
      const { text: aiText, projects, resources, allocation, gantt, heatmap } = res.data || {};

      // Show the AI text
      setMessages((prev) => [...prev, { sender: "ai", text: aiText || "(no response)" }]);

      // Prefer backend-provided charts if present; otherwise recompute from payloads
      if (Array.isArray(gantt) || Array.isArray(heatmap)) {
        setAiData((prev) => ({
          gantt: Array.isArray(gantt) ? gantt : prev.gantt,
          heatmap: Array.isArray(heatmap) ? heatmap : prev.heatmap,
        }));
      } else {
        // Optional recompute if backend didn’t send charts but sent data
        if (Array.isArray(projects) || Array.isArray(resources)) {
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
          <h2>Results / Visualizations</h2>

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
