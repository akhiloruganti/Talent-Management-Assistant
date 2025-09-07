import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css"; // Import custom CSS

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/">Talent Management</Link>
      </div>
      <ul className="navbar-links">
        
        <li><Link to="/projects">Projects</Link></li>
        <li><Link to="/resources">Resources</Link></li>
        <li><Link to="/chatbot">Chatbot</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;
