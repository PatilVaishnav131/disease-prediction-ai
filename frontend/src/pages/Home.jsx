import React from "react";
import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import medicalHero from "../assets/images/medical-hero.jpg";

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white flex flex-col">
      <Navbar />

      {/* Hero Section */}
      <section className="flex flex-col md:flex-row items-center justify-between px-10 py-20">
        <div className="md:w-1/2">
          <h2 className="text-4xl font-extrabold text-blue-800 mb-4">
            AI-Powered Disease Prediction
          </h2>
          <p className="text-gray-700 mb-6">
            Upload your reports or answer simple questions to assess your health risks using our AI models.
          </p>
          <Link
            to="/predict"
            className="inline-block bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg shadow-md transition"
          >
            Start Prediction
          </Link>
        </div>
        <div className="md:w-1/2 mt-10 md:mt-0">
        <img
            src={medicalHero}
            alt="Medical AI"
            className="w-full max-w-md mx-auto rounded-xl shadow-lg"
            />

        </div>
      </section>

      {/* Features Section */}
      <section className="bg-white py-16 px-6">
        <h3 className="text-3xl text-center font-bold text-blue-700 mb-10">What We Offer</h3>
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="p-6 border rounded-lg shadow hover:shadow-md transition">
            <h4 className="text-xl font-semibold mb-2 text-blue-600">Disease Prediction</h4>
            <p className="text-gray-600">Use AI to predict heart disease, diabetes, and more.</p>
          </div>
          <div className="p-6 border rounded-lg shadow hover:shadow-md transition">
            <h4 className="text-xl font-semibold mb-2 text-blue-600">Symptom Checker</h4>
            <p className="text-gray-600">Answer a few questions for a quick assessment.</p>
          </div>
          <div className="p-6 border rounded-lg shadow hover:shadow-md transition">
            <h4 className="text-xl font-semibold mb-2 text-blue-600">Health Education</h4>
            <p className="text-gray-600">Explore articles and expert advice to stay healthy.</p>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Home;
