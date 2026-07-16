import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import NewGlasses from "../newglasses.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <NewGlasses />
  </StrictMode>,
);
