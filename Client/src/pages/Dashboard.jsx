// import React, { useEffect, useState } from "react";

// import API from "../api/axios"; 
// import Navbar from "../components/Navbar";
// const Dashboard = () => {
//   const [allocations, setAllocations] = useState([]);

//   useEffect(() => {
//     API.get("/allocations")
//       .then(res => setAllocations(res.data))
//       .catch(err => console.error(err));
//   }, []);

//   return (
//     <div>
//       <Navbar />
//       <div className="container">
//         <h1>Dashboard</h1>
       
//         <p>Gantt chart, heatmaps, and KPIs will go here.</p>
//       </div>
//     </div>
//   );
// };

// export default Dashboard;
