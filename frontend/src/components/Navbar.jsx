import { Link } from "react-router-dom";

const Navbar = () => (
  <header className="bg-white shadow p-6 flex justify-between items-center">
    <h1 className="text-2xl font-bold text-blue-700">MediPredict</h1>
    <nav className="space-x-6 text-blue-600 font-medium">
      <Link to="/">Home</Link>
      <Link to="/predict">Predict</Link>
      <Link to="/about">About</Link>
    </nav>
  </header>
);

export default Navbar;
