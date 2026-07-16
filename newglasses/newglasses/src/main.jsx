// main.jsx
import React from 'react'; // Add this line
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import NewGlasses from "../newglasses.jsx"; // Adjust path if needed

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <NewGlasses />
  </StrictMode>,
);
