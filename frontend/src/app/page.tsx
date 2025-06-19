'use client';

import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { apexVerifyAPI, AnalysisResult, EvolutionStatus } from '@/lib/api';
import { Upload, Shield, Brain, TrendingUp, Download, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [evolutionStatus, setEvolutionStatus] = useState<EvolutionStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [userFeedback, setUserFeedback] = useState<boolean | null>(null);
  const [useEvolution, setUseEvolution] = useState(true);
  const [isTriggeringEvolution, setIsTriggeringEvolution] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load evolution status on component mount
  useState(() => {
    loadEvolutionStatus();
  });

  const loadEvolutionStatus = async () => {
    try {
      const status = await apexVerifyAPI.getEvolutionStatus();
      setEvolutionStatus(status);
    } catch (error) {
      console.error('Failed to load evolution status:', error);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setAnalysisResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const result = await apexVerifyAPI.analyzeFile(file, userFeedback || undefined, useEvolution);
      setAnalysisResult(result);
      
      // Reload evolution status after analysis
      await loadEvolutionStatus();
    } catch (error) {
      setError('Analysis failed. Please try again.');
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFeedback = async (feedback: boolean) => {
    if (!analysisResult) return;

    try {
      await apexVerifyAPI.submitFeedback({
        file_id: 'temp-id', // In a real app, you'd get this from the analysis result
        user_feedback: feedback,
      });
      
      setUserFeedback(feedback);
      setError(null);
    } catch (error) {
      setError('Failed to submit feedback');
      console.error('Feedback error:', error);
    }
  };

  const handleTriggerEvolution = async () => {
    setIsTriggeringEvolution(true);
    try {
      const result = await apexVerifyAPI.triggerEvolution(10); // Lower threshold for demo
      if (result.triggered) {
        setError(null);
        // Reload status after evolution
        await loadEvolutionStatus();
      } else {
        setError(`Evolution not triggered. Need ${result.data_points_available} more data points.`);
      }
    } catch (error) {
      setError('Failed to trigger evolution');
      console.error('Evolution error:', error);
    } finally {
      setIsTriggeringEvolution(false);
    }
  };

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const getPredictionColor = (prediction: string) => {
    return prediction === 'REAL' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Shield className="h-12 w-12 text-blue-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-900">ApexVerify AI</h1>
          </div>
          <p className="text-xl text-gray-600 mb-2">The New Standard for Authenticity in the Creator Economy</p>
          <div className="flex items-center justify-center space-x-2">
            <Badge variant="secondary" className="bg-blue-100 text-blue-800">
              <Brain className="h-4 w-4 mr-1" />
              OpenEvolve AI
            </Badge>
            <Badge variant="secondary" className="bg-green-100 text-green-800">
              <TrendingUp className="h-4 w-4 mr-1" />
              Self-Learning
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Analysis Card */}
          <div className="lg:col-span-2">
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Upload className="h-5 w-5 mr-2" />
                  Deepfake Analysis
                </CardTitle>
                <CardDescription>
                  Upload an image or video to analyze for deepfake detection using our OpenEvolve AI
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* File Upload */}
                <div className="space-y-2">
                  <Label htmlFor="file">Select Media File</Label>
                  <Input
                    id="file"
                    type="file"
                    accept="image/*,video/*"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                    className="cursor-pointer"
                  />
                  {file && (
                    <p className="text-sm text-gray-600">
                      Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                  )}
                </div>

                {/* Evolution Toggle */}
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="useEvolution"
                    checked={useEvolution}
                    onChange={(e) => setUseEvolution(e.target.checked)}
                    className="rounded"
                  />
                  <Label htmlFor="useEvolution" className="text-sm">
                    Use OpenEvolve AI (Self-Learning)
                  </Label>
                </div>

                {/* Analyze Button */}
                <Button
                  onClick={handleAnalyze}
                  disabled={!file || isAnalyzing}
                  className="w-full"
                  size="lg"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Shield className="h-4 w-4 mr-2" />
                      Analyze for Deepfakes
                    </>
                  )}
                </Button>

                {/* Error Display */}
                {error && (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {/* Analysis Result */}
                {analysisResult && (
                  <Card className="border-2">
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <span>Analysis Result</span>
                        <Badge className={getPredictionColor(analysisResult.prediction)}>
                          {analysisResult.prediction}
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label className="text-sm font-medium">Confidence</Label>
                          <div className="flex items-center space-x-2">
                            <Progress value={analysisResult.confidence * 100} className="flex-1" />
                            <span className="text-sm font-medium">
                              {formatConfidence(analysisResult.confidence)}
                            </span>
                          </div>
                        </div>
                        <div>
                          <Label className="text-sm font-medium">Faces Detected</Label>
                          <p className="text-lg font-semibold">{analysisResult.faces_detected}</p>
                        </div>
                      </div>

                      {/* Evolution Data */}
                      {analysisResult.evolution_data && (
                        <div className="bg-blue-50 p-4 rounded-lg">
                          <h4 className="font-medium text-blue-900 mb-2">OpenEvolve AI Data</h4>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-blue-700">Evolution Used:</span>
                              <span className="ml-2">
                                {analysisResult.evolution_data.used_evolved_program ? (
                                  <CheckCircle className="h-4 w-4 text-green-600 inline" />
                                ) : (
                                  <XCircle className="h-4 w-4 text-red-600 inline" />
                                )}
                              </span>
                            </div>
                            <div>
                              <span className="text-blue-700">Evolution Accuracy:</span>
                              <span className="ml-2 font-medium">
                                {formatConfidence(analysisResult.evolution_data.evolution_accuracy)}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Learning Metrics */}
                      {analysisResult.learning_metrics && (
                        <div className="bg-green-50 p-4 rounded-lg">
                          <h4 className="font-medium text-green-900 mb-2">Learning Metrics</h4>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-green-700">Learning Confidence:</span>
                              <span className="ml-2 font-medium">
                                {formatConfidence(analysisResult.learning_metrics.confidence)}
                              </span>
                            </div>
                            {analysisResult.learning_metrics.accuracy && (
                              <div>
                                <span className="text-green-700">Learning Accuracy:</span>
                                <span className="ml-2 font-medium">
                                  {formatConfidence(analysisResult.learning_metrics.accuracy)}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Feedback Section */}
                      <div className="space-y-2">
                        <Label className="text-sm font-medium">Was this prediction correct?</Label>
                        <div className="flex space-x-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleFeedback(true)}
                            className="flex-1"
                          >
                            <CheckCircle className="h-4 w-4 mr-1" />
                            Correct
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleFeedback(false)}
                            className="flex-1"
                          >
                            <XCircle className="h-4 w-4 mr-1" />
                            Incorrect
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Evolution Status Sidebar */}
          <div className="space-y-6">
            {/* Evolution Status Card */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Brain className="h-5 w-5 mr-2" />
                  OpenEvolve Status
                </CardTitle>
                <CardDescription>
                  Real-time learning and evolution statistics
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {evolutionStatus ? (
                  <>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <Label className="text-xs text-gray-600">Total Analyses</Label>
                        <p className="text-lg font-semibold">{evolutionStatus.learning_stats.total_analyses}</p>
                      </div>
                      <div>
                        <Label className="text-xs text-gray-600">Evolution Iterations</Label>
                        <p className="text-lg font-semibold">{evolutionStatus.learning_stats.evolution_iterations}</p>
                      </div>
                      <div>
                        <Label className="text-xs text-gray-600">Best Accuracy</Label>
                        <p className="text-lg font-semibold text-green-600">
                          {formatConfidence(evolutionStatus.learning_stats.best_accuracy)}
                        </p>
                      </div>
                      <div>
                        <Label className="text-xs text-gray-600">Data Points</Label>
                        <p className="text-lg font-semibold">{evolutionStatus.analysis_history_size}</p>
                      </div>
                    </div>

                    <Button
                      onClick={handleTriggerEvolution}
                      disabled={isTriggeringEvolution}
                      variant="outline"
                      className="w-full"
                    >
                      {isTriggeringEvolution ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2" />
                          Triggering Evolution...
                        </>
                      ) : (
                        <>
                          <TrendingUp className="h-4 w-4 mr-2" />
                          Trigger Evolution
                        </>
                      )}
                    </Button>
                  </>
                ) : (
                  <p className="text-sm text-gray-500">Loading evolution status...</p>
                )}
              </CardContent>
            </Card>

            {/* Quick Actions Card */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline" className="w-full justify-start">
                      <Download className="h-4 w-4 mr-2" />
                      Export Learning Data
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Export Learning Data</DialogTitle>
                      <DialogDescription>
                        Download the complete learning history and evolution data for analysis.
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <p className="text-sm text-gray-600">
                        This will export all analysis history, feedback data, and evolution statistics.
                      </p>
                      <Button className="w-full">
                        <Download className="h-4 w-4 mr-2" />
                        Download JSON Export
                      </Button>
                    </div>
                  </DialogContent>
                </Dialog>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600">
          <p className="text-sm">
            ApexVerify AI with OpenEvolve - Bringing self-learning AI to deepfake detection
          </p>
          <p className="text-xs mt-2">
            Powered by OpenEvolve and AlphaEvolve research
          </p>
        </div>
      </div>
    </div>
  );
}
