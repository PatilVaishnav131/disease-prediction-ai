import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import './index.css';
import MedicalPredictionPage from "./components/Predict";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predict" element={<MedicalPredictionPage/>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
