import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import ResultDisplay from './components/ResultDisplay';
import { Brain, Upload, MessageSquare } from 'lucide-react';

interface AnalysisResult {
  success: boolean;
  message?: string;
  data?: {
    summary: string;
    confidence: number;
    results: {
      vqa?: {
        answer: string;
        confidence: number;
      };
      object_detection?: {
        detections: Array<{
          label: string;
          confidence: number;
        }>;
        count: number;
      };
      document_qa?: {
        answer: string;
        confidence: number;
      };
      image_description?: {
        description: string;
      };
    };
  };
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<File | null>(null);
  const [selectedAudio, setSelectedAudio] = useState<File | null>(null);
  const [question, setQuestion] = useState('');
  const [detectionLabels, setDetectionLabels] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!selectedImage && !selectedDocument && !selectedAudio && !question.trim()) {
      alert('Please provide at least one input (image, document, audio, or question)');
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    
    if (selectedImage) formData.append('image', selectedImage);
    if (selectedDocument) formData.append('document', selectedDocument);
    if (selectedAudio) formData.append('audio', selectedAudio);
    if (question.trim()) formData.append('question', question.trim());
    if (detectionLabels.trim()) formData.append('detection_labels', detectionLabels.trim());

    try {
      const response = await axios.post<AnalysisResult>(`${API_BASE_URL}/analyze/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      });
      setResult(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      setResult({
        success: false,
        message: axios.isAxiosError(error) 
          ? error.response?.data?.message || 'Analysis failed. Please try again.'
          : 'Analysis failed. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setSelectedDocument(null);
    setSelectedAudio(null);
    setQuestion('');
    setDetectionLabels('');
    setResult(null);
  };

  const imageTypes = {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
    'image/gif': ['.gif']
  };

  const documentTypes = {
    'application/pdf': ['.pdf'],
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png']
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-600" />
              <h1 className="ml-2 text-xl font-bold text-gray-900">
                OmniSense AI
              </h1>
            </div>
            <p className="text-sm text-gray-500">
              Multimodal AI Platform
            </p>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h2 className="text-lg font-semibold mb-4 flex items-center">
                <Upload className="h-5 w-5 mr-2" />
                Upload Files
              </h2>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Image
                </label>
                {selectedImage ? (
                  <div className="flex items-center justify-between bg-gray-50 p-3 rounded">
                    <span className="text-sm">{selectedImage.name}</span>
                    <button
                      onClick={() => setSelectedImage(null)}
                      className="text-red-500 hover:text-red-700"
                    >
                      Remove
                    </button>
                  </div>
                ) : (
                  <FileUpload
                    onFileSelect={setSelectedImage}
                    acceptedTypes={imageTypes}
                    label="image"
                  />
                )}
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Document
                </label>
                {selectedDocument ? (
                  <div className="flex items-center justify-between bg-gray-50 p-3 rounded">
                    <span className="text-sm">{selectedDocument.name}</span>
                    <button
                      onClick={() => setSelectedDocument(null)}
                      className="text-red-500 hover:text-red-700"
                    >
                      Remove
                    </button>
                  </div>
                ) : (
                  <FileUpload
                    onFileSelect={setSelectedDocument}
                    acceptedTypes={documentTypes}
                    label="document"
                  />
                )}
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Audio (Coming Soon)
                </label>
                {selectedAudio ? (
                  <div className="flex items-center justify-between bg-gray-50 p-3 rounded">
                    <span className="text-sm">{selectedAudio.name}</span>
                    <button
                      onClick={() => setSelectedAudio(null)}
                      className="text-red-500 hover:text-red-700"
                    >
                      Remove
                    </button>
                  </div>
                ) : (
                  <div className="border-2 border-dashed border-gray-200 rounded-lg p-6 text-center">
                    <Upload className="mx-auto h-12 w-12 text-gray-300 mb-4" />
                    <p className="text-sm text-gray-400">Audio processing coming soon</p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <h2 className="text-lg font-semibold mb-4 flex items-center">
                <MessageSquare className="h-5 w-5 mr-2" />
                Questions & Queries
              </h2>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Question (Optional)
                </label>
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question about your uploaded content..."
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  rows={3}
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Object Detection Labels (Optional)
                </label>
                <input
                  type="text"
                  value={detectionLabels}
                  onChange={(e) => setDetectionLabels(e.target.value)}
                  placeholder="person, car, dog, cat (comma-separated)"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Leave empty for default object detection
                </p>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className={`flex-1 py-2 px-4 rounded-md font-medium ${
                    loading
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700'
                  } text-white`}
                >
                  {loading ? 'Analyzing...' : 'Analyze'}
                </button>
                <button
                  onClick={handleReset}
                  className="py-2 px-4 border border-gray-300 rounded-md font-medium text-gray-700 hover:bg-gray-50"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-lg font-semibold mb-4">Analysis Results</h2>
            {result || loading ? (
              <ResultDisplay result={result} loading={loading} />
            ) : (
              <div className="bg-white p-8 rounded-lg shadow-md text-center">
                <Brain className="mx-auto h-16 w-16 text-gray-300 mb-4" />
                <p className="text-gray-500">
                  Upload files and click "Analyze" to see results
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;