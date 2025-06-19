# ApexVerify AI with OpenEvolve Integration

## Overview

ApexVerify AI has been enhanced with **OpenEvolve**, an open-source implementation of AlphaEvolve that provides self-learning capabilities through reinforcement learning and evolutionary algorithms. This integration enables the deepfake detection system to continuously improve its accuracy based on user feedback and real-world data.

## What is OpenEvolve?

[OpenEvolve](https://github.com/codelion/openevolve) is an open-source implementation of AlphaEvolve, a system that uses Large Language Models (LLMs) to evolve and improve code through iterative refinement. It combines:

- **Evolutionary Algorithms**: Genetic programming principles to evolve detection logic
- **Reinforcement Learning**: Learning from user feedback and ground truth data
- **LLM Integration**: Using advanced language models to generate improved code
- **Self-Learning**: Continuous improvement without manual intervention

## Key Features

### 🧠 Self-Learning Deepfake Detection
- **Continuous Evolution**: The detection algorithm evolves based on user feedback
- **Adaptive Thresholds**: Confidence thresholds adjust automatically
- **Pattern Recognition**: Learns new deepfake patterns over time
- **User Feedback Integration**: Incorporates human expertise into the learning process

### 🔄 Evolutionary Algorithm
- **Population-based Learning**: Maintains multiple detection strategies
- **Fitness-based Selection**: Best performing algorithms are preserved and improved
- **Mutation and Crossover**: Generates new detection approaches
- **Checkpoint System**: Saves progress and allows rollback to previous versions

### 📊 Learning Analytics
- **Real-time Metrics**: Track accuracy improvements over time
- **Evolution History**: View how the detection algorithm has evolved
- **Performance Insights**: Analyze learning patterns and trends
- **Export Capabilities**: Export learning data for external analysis

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Upload   │───▶│  Feature Extract │───▶│ OpenEvolve     │
│   (Image/Video) │    │  (512-dim vector)│    │  Detector       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Analysis Result │    │  Evolution      │
                       │  + Confidence    │    │  Engine         │
                       │  + Prediction    │    │  (LLM-based)    │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  User Feedback   │───▶│  Learning       │
                       │  (Real/Fake)     │    │  Update         │
                       └──────────────────┘    └─────────────────┘
```

## API Endpoints

### Core Analysis Endpoints

#### `POST /analyze`
Analyze media files with OpenEvolve integration

**Parameters:**
- `file`: Media file (image/video)
- `user_feedback`: Optional boolean (true for fake, false for real)
- `use_evolution`: Boolean (default: true) - Enable OpenEvolve analysis

**Response:**
```json
{
  "is_fake": true,
  "confidence": 0.87,
  "prediction": "FAKE",
  "faces_detected": 1,
  "media_type": "image",
  "learning_metrics": {
    "confidence": 0.87,
    "accuracy": 0.92
  },
  "evolution_data": {
    "used_evolved_program": true,
    "evolution_accuracy": 0.89,
    "program_version": "evolved"
  }
}
```

### Evolution Management Endpoints

#### `GET /evolution/status`
Get current evolution status and statistics

**Response:**
```json
{
  "evolution_enabled": true,
  "openevolve_available": true,
  "learning_stats": {
    "total_analyses": 150,
    "evolution_iterations": 3,
    "best_accuracy": 0.94,
    "accuracy_improvements": [...]
  },
  "analysis_history_size": 150,
  "feedback_history_size": 45,
  "evolution_directory": "evolution_output",
  "recent_evolutions": [...]
}
```

#### `POST /evolution/trigger`
Manually trigger evolution process

**Parameters:**
- `min_data_points`: Minimum data points required (default: 50)

**Response:**
```json
{
  "status": "success",
  "triggered": true,
  "data_points_available": 150
}
```

#### `GET /evolution/export`
Export learning data for analysis

**Response:**
```json
{
  "learning_stats": {...},
  "analysis_history": [...],
  "feedback_history": [...],
  "evolution_status": {...},
  "export_timestamp": "2024-01-15T10:30:00"
}
```

### Feedback Endpoints

#### `POST /feedback`
Submit user feedback for learning

**Parameters:**
- `file_id`: File identifier
- `user_feedback`: Boolean (true for fake, false for real)
- `confidence_rating`: Optional float (0.0-1.0)

**Response:**
```json
{
  "status": "success",
  "message": "Feedback submitted successfully"
}
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- FastAPI
- OpenEvolve (automatically installed)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ApexverifyAi
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install OpenEvolve:**
```bash
pip install git+https://github.com/codelion/openevolve.git
```

4. **Run the server:**
```bash
python run_server.py
```

### Configuration

The OpenEvolve integration can be configured through environment variables:

```bash
# Enable/disable evolution
export EVOLUTION_ENABLED=true

# Evolution directory
export EVOLUTION_DIR=evolution_output

# Minimum data points for evolution
export MIN_DATA_POINTS=50

# Evolution checkpoint interval
export CHECKPOINT_INTERVAL=10
```

## Usage Examples

### Basic Analysis with Evolution

```python
import requests

# Upload and analyze with evolution
files = {'file': open('test_image.jpg', 'rb')}
response = requests.post(
    'http://localhost:8000/analyze?use_evolution=true',
    files=files
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Evolution used: {result['evolution_data']['used_evolved_program']}")
```

### Submit Feedback for Learning

```python
# Submit feedback to improve the model
feedback_data = {
    'file_id': 'abc123',
    'user_feedback': True,  # This was actually fake
    'confidence_rating': 0.9
}
response = requests.post('http://localhost:8000/feedback', json=feedback_data)
```

### Monitor Evolution Status

```python
# Check evolution status
response = requests.get('http://localhost:8000/evolution/status')
status = response.json()
print(f"Total analyses: {status['learning_stats']['total_analyses']}")
print(f"Best accuracy: {status['learning_stats']['best_accuracy']}")
```

### Trigger Manual Evolution

```python
# Trigger evolution when enough data is available
response = requests.post('http://localhost:8000/evolution/trigger?min_data_points=50')
result = response.json()
if result['triggered']:
    print("Evolution triggered successfully!")
else:
    print(f"Insufficient data. Need {result['data_points_available']} more points.")
```

## Learning Process

### 1. Initial Detection
- System starts with basic threshold-based detection
- Extracts 512-dimensional feature vectors from images
- Makes initial predictions with confidence scores

### 2. User Feedback Collection
- Users provide feedback on detection accuracy
- System stores feedback with metadata
- Builds training dataset for evolution

### 3. Evolution Triggering
- When sufficient data is collected (default: 50 points)
- OpenEvolve generates improved detection algorithms
- Uses LLM to evolve code based on feedback patterns

### 4. Algorithm Evolution
- **Selection**: Best performing algorithms are preserved
- **Mutation**: Small changes to detection logic
- **Crossover**: Combine successful strategies
- **Evaluation**: Test new algorithms on historical data

### 5. Deployment
- Best evolved algorithm becomes active
- System continues learning with new algorithm
- Process repeats for continuous improvement

## Performance Metrics

### Accuracy Tracking
- **Baseline Accuracy**: Initial detection performance
- **Evolution Accuracy**: Performance after evolution
- **Improvement Rate**: Rate of accuracy gains over time

### Learning Efficiency
- **Data Efficiency**: How quickly accuracy improves with data
- **Evolution Frequency**: How often evolution occurs
- **Convergence Time**: Time to reach target accuracy

### User Experience
- **Feedback Response Time**: How quickly feedback is incorporated
- **Prediction Confidence**: Reliability of confidence scores
- **False Positive/Negative Rates**: Error rates over time

## Monitoring and Analytics

### Real-time Dashboard
Access the web interface at `http://localhost:8000` to view:
- Current evolution status
- Learning statistics
- Recent predictions
- Accuracy trends

### API Monitoring
Use the `/evolution/status` endpoint to monitor:
- Total analyses performed
- Evolution iterations completed
- Best accuracy achieved
- Data collection progress

### Export Capabilities
Use `/evolution/export` to export:
- Complete learning history
- Evolution checkpoints
- Performance metrics
- User feedback data

## Troubleshooting

### Common Issues

1. **Evolution Not Triggering**
   - Check if enough data points are collected
   - Verify evolution is enabled
   - Check logs for error messages

2. **Low Accuracy**
   - Ensure diverse training data
   - Check user feedback quality
   - Monitor evolution progress

3. **Performance Issues**
   - Reduce evolution frequency
   - Limit analysis history size
   - Use lighter model configurations

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python run_server.py
```

### Reset Learning

To reset the learning process:
```bash
rm -rf evolution_output/*
rm -rf data/feedback.json
```

## Future Enhancements

### Planned Features
- **Multi-modal Learning**: Support for video and audio analysis
- **Advanced Evolution**: More sophisticated genetic algorithms
- **Distributed Learning**: Multi-node evolution for faster convergence
- **Real-time Evolution**: Continuous evolution without checkpoints

### Research Integration
- **Academic Benchmarks**: Integration with standard datasets
- **Comparative Analysis**: Compare with other deepfake detection methods
- **Publication Support**: Export results for research papers

## Contributing

We welcome contributions to improve the OpenEvolve integration:

1. **Bug Reports**: Report issues with detailed descriptions
2. **Feature Requests**: Suggest new capabilities
3. **Code Contributions**: Submit pull requests with improvements
4. **Documentation**: Help improve documentation and examples

## License

This project is licensed under the Apache 2.0 License. OpenEvolve is also licensed under Apache 2.0.

## Acknowledgments

- **OpenEvolve Team**: For the excellent open-source implementation
- **AlphaEvolve Research**: For the foundational research
- **Community Contributors**: For feedback and improvements

## Support

For support and questions:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the comprehensive documentation
- **Community**: Join our community discussions

---

**ApexVerify AI with OpenEvolve** - Bringing self-learning AI to deepfake detection! 🚀 