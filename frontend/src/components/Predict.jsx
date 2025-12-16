import { useState } from 'react';

const MedicalPredictionPage = () => {
  const [selectedOption, setSelectedOption] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const detectionOptions = [
    {
      id: 'malaria',
      title: 'Malaria Detection',
      description: 'Analyze blood smear images for malaria parasites',
      icon: '🦟',
      apiEndpoint: 'http://127.0.0.1:8000/predict/malaria',
      color: 'from-red-500 to-pink-500'
    },
    {
      id: 'skin',
      title: 'Skin Disease Detection',
      description: 'Identify various skin conditions and diseases',
      icon: '🔬',
      apiEndpoint: 'http://127.0.0.1:8000/predict/skin',
      color: 'from-orange-500 to-amber-500'
    },
    {
      id: 'brain',
      title: 'Brain Tumor Detection',
      description: 'Detect and classify brain tumors from MRI scans',
      icon: '🧠',
      apiEndpoint: 'http://127.0.0.1:8000/predict/brain',
      color: 'from-purple-500 to-indigo-500'
    }
  ];

  const handleOptionSelect = (option) => {
    setSelectedOption(option);
    setUploadedImage(null);
    setImagePreview(null);
    setResult(null);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  const handlePredict = async () => {
    if (!uploadedImage || !selectedOption) return;

    setIsLoading(true);
    const formData = new FormData();

    // IMPORTANT: backend expects "file"
    formData.append('file', uploadedImage);

    try {
      const response = await fetch(selectedOption.apiEndpoint, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Prediction error:', error);
      alert("Prediction failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };


  const resetAll = () => {
    setSelectedOption(null);
    setUploadedImage(null);
    setImagePreview(null);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-800 mb-4">
            Medical Image Analysis
          </h1>
          <p className="text-xl text-gray-600">
            AI-powered disease detection from medical imagery
          </p>
        </div>

        {!selectedOption ? (
          /* Selection Grid */
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {detectionOptions.map((option) => (
              <div
                key={option.id}
                onClick={() => handleOptionSelect(option)}
                className="bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer transform hover:scale-105 overflow-hidden"
              >
                <div className={`h-32 bg-gradient-to-br ${option.color} flex items-center justify-center`}>
                  <span className="text-7xl">{option.icon}</span>
                </div>
                <div className="p-6">
                  <h3 className="text-2xl font-bold text-gray-800 mb-3">
                    {option.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {option.description}
                  </p>
                  <button className="mt-6 w-full bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 transition-all duration-300">
                    Select
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* Upload and Analysis Section */
          <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
            {/* Header Bar */}
            <div className={`bg-gradient-to-r ${selectedOption.color} p-6 flex items-center justify-between`}>
              <div className="flex items-center gap-4">
                <span className="text-5xl">{selectedOption.icon}</span>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {selectedOption.title}
                  </h2>
                  <p className="text-white text-opacity-90">
                    {selectedOption.description}
                  </p>
                </div>
              </div>
              <button
                onClick={resetAll}
                className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-6 py-2 rounded-lg font-semibold transition-all duration-300"
              >
                ← Back
              </button>
            </div>

            <div className="p-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Upload Section */}
                <div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-4">
                    Upload Image
                  </h3>
                  
                  {!imagePreview ? (
                    <label className="block">
                      <div className="border-4 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-blue-500 transition-all duration-300 cursor-pointer bg-gray-50 hover:bg-blue-50">
                        <div className="text-6xl mb-4">📁</div>
                        <p className="text-xl font-semibold text-gray-700 mb-2">
                          Click to upload image
                        </p>
                        <p className="text-gray-500">
                          PNG, JPG, JPEG up to 10MB
                        </p>
                      </div>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                      />
                    </label>
                  ) : (
                    <div className="space-y-4">
                      <div className="relative rounded-xl overflow-hidden border-4 border-gray-200">
                        <img
                          src={imagePreview}
                          alt="Uploaded preview"
                          className="w-full h-96 object-contain bg-gray-100"
                        />
                      </div>
                      <div className="flex gap-4">
                        <label className="flex-1 cursor-pointer">
                          <div className="w-full bg-gray-200 hover:bg-gray-300 text-gray-800 py-3 rounded-lg font-semibold transition-all duration-300 text-center">
                            Change Image
                          </div>
                          <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageUpload}
                            className="hidden"
                          />
                        </label>
                        <button
                          onClick={handlePredict}
                          disabled={isLoading}
                          className={`flex-1 bg-gradient-to-r ${selectedOption.color} text-white py-3 rounded-lg font-semibold transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg`}
                        >
                          {isLoading ? (
                            <span className="flex items-center justify-center gap-2">
                              <span className="animate-spin">⚙️</span>
                              Analyzing...
                            </span>
                          ) : (
                            'Analyze Image'
                          )}
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Results Section */}
                <div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-4">
                    Analysis Results
                  </h3>
                  
                  {!result && !isLoading && (
                    <div className="bg-gray-50 rounded-xl p-12 text-center border-2 border-gray-200 h-96 flex flex-col items-center justify-center">
                      <div className="text-6xl mb-4">📊</div>
                      <p className="text-gray-500 text-lg">
                        Upload an image and click "Analyze Image" to see results
                      </p>
                    </div>
                  )}

                  {isLoading && (
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-12 text-center border-2 border-blue-200 h-96 flex flex-col items-center justify-center">
                      <div className="text-6xl mb-4 animate-bounce">🔍</div>
                      <p className="text-gray-700 text-xl font-semibold">
                        Analyzing your image...
                      </p>
                      <div className="mt-6 flex gap-2">
                        <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                        <div className="w-3 h-3 bg-indigo-500 rounded-full animate-pulse delay-100"></div>
                        <div className="w-3 h-3 bg-purple-500 rounded-full animate-pulse delay-200"></div>
                      </div>
                    </div>
                  )}

                  {result && !isLoading && (
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-8 border-2 border-green-200">
                      <div className="text-center mb-6">
                        <div className="text-6xl mb-4">✅</div>
                        <h4 className="text-2xl font-bold text-gray-800">
                          Analysis Complete
                        </h4>
                      </div>
                      
                      <div className="space-y-4">
                        <div className="bg-white rounded-lg p-4 shadow">
                          <p className="text-sm text-gray-600 mb-1">Prediction</p>
                          <p className="text-xl font-bold text-gray-800">
                            {result.prediction}
                          </p>
                        </div>
                        
                        <div className="bg-white rounded-lg p-4 shadow">
                          <p className="text-sm text-gray-600 mb-1">Confidence</p>
                          <p className="text-xl font-bold text-green-600">
                            {(result.confidence * 100).toFixed(2)}%
                          </p>

                        </div>
                        
                        <div className="bg-white rounded-lg p-4 shadow">
                          <p className="text-sm text-gray-600 mb-1">Details</p>
                          <p className="text-gray-700">
                            {result.details || "AI-based medical image analysis completed."}
                          </p>

                        </div>
                      </div>
                      
                      <button
                        onClick={() => {
                          setUploadedImage(null);
                          setImagePreview(null);
                          setResult(null);
                        }}
                        className="mt-6 w-full bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 transition-all duration-300"
                      >
                        Analyze Another Image
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer Info */}
        <div className="mt-12 text-center text-gray-600">
          <p className="text-sm">
            ⚕️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
          </p>
        </div>
      </div>
    </div>
  );
};

export default MedicalPredictionPage;