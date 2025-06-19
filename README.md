# ApexVerify AI with OpenEvolve Integration

## 🛡️ The New Standard for Authenticity in the Creator Economy

**ApexVerify AI** is more than a project—it's a brand. We empower creators and audiences to navigate the digital world with confidence by verifying the authenticity of content. Our mission is to make authenticity a badge of honor, not just a checkbox.

## 🚀 What's New in v2.0

This version introduces **OpenEvolve integration**, bringing self-learning AI capabilities to deepfake detection:

- **🧠 Self-Learning AI**: Powered by OpenEvolve (AlphaEvolve implementation)
- **🔄 Continuous Evolution**: Detection algorithms improve over time
- **📊 Real-time Analytics**: Monitor learning progress and accuracy improvements
- **🎯 User Feedback Integration**: Human expertise enhances AI learning
- **⚡ Modern Next.js Frontend**: Beautiful, responsive interface

## 🌟 Key Features

### Core Deepfake Detection
- **AI-powered Analysis**: Advanced deepfake detection using state-of-the-art models
- **Multi-format Support**: Images and videos
- **Real-time Processing**: Fast analysis with confidence scores
- **Face Detection**: Automatic face detection and analysis

### OpenEvolve Self-Learning
- **Evolutionary Algorithms**: Genetic programming for algorithm improvement
- **Reinforcement Learning**: Learning from user feedback and ground truth
- **LLM Integration**: Using advanced language models for code evolution
- **Checkpoint System**: Save and restore learning progress

### Creator Tools
- **Content Verification**: Verify authenticity of creator content
- **Trust Badges**: Generate and apply trust watermarks
- **Metadata Integration**: Embed verification data in content
- **Public Verification**: Lookup and validate verified content

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js       │───▶│  FastAPI         │───▶│  OpenEvolve     │
│   Frontend      │    │  Backend         │    │  AI Engine      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User          │    │  Deepfake        │    │  Evolution      │
│   Interface     │    │  Detection       │    │  Engine         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core application logic
- **FastAPI**: High-performance web framework
- **OpenEvolve**: Self-learning AI integration
- **PyTorch**: Deep learning models
- **OpenCV**: Computer vision processing

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Beautiful component library
- **Lucide React**: Icon library

### AI & ML
- **OpenEvolve**: AlphaEvolve implementation
- **Reinforcement Learning**: User feedback integration
- **Evolutionary Algorithms**: Genetic programming
- **Feature Extraction**: 512-dimensional vectors

## 📦 Installation

### Prerequisites
- Node.js 18.0.0+
- Python 3.8+
- Git

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/urban7733/ApexVerify.git
cd ApexVerify
```

2. **Install all dependencies:**
```bash
npm run install:all
```

3. **Start the development servers:**
```bash
npm run dev
```

This will start both the backend (port 8000) and frontend (port 3000) simultaneously.

### Manual Installation

#### Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install OpenEvolve
pip install git+https://github.com/codelion/openevolve.git

# Start backend
python run_server.py
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🎯 Usage

### Web Interface
1. Open [http://localhost:3000](http://localhost:3000)
2. Upload an image or video file
3. Enable OpenEvolve AI for self-learning analysis
4. View results with confidence scores and learning metrics
5. Provide feedback to improve the AI

### API Endpoints

#### Core Analysis
- `POST /analyze` - Analyze media files with OpenEvolve
- `POST /upload` - Upload files for analysis
- `GET /api/health` - Health check

#### Evolution Management
- `GET /evolution/status` - Get evolution status
- `POST /evolution/trigger` - Trigger evolution process
- `GET /evolution/export` - Export learning data

#### Feedback
- `POST /feedback` - Submit user feedback for learning

### Example API Usage

```python
import requests

# Analyze with OpenEvolve
files = {'file': open('test_image.jpg', 'rb')}
response = requests.post(
    'http://localhost:8000/analyze?use_evolution=true',
    files=files
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend Configuration

The OpenEvolve integration can be configured through environment variables:

```bash
# Enable/disable evolution
export EVOLUTION_ENABLED=true

# Evolution directory
export EVOLUTION_DIR=evolution_output

# Minimum data points for evolution
export MIN_DATA_POINTS=50
```

## 📊 Learning Process

### 1. Initial Detection
- System starts with basic threshold-based detection
- Extracts 512-dimensional feature vectors
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

## 🧪 Testing

### Run All Tests
```bash
npm test
```

### Backend Tests
```bash
python -m pytest test_openevolve_integration.py -v
```

### Frontend Tests
```bash
npm run test:frontend
```

## 📈 Performance Metrics

### Accuracy Tracking
- **Baseline Accuracy**: Initial detection performance
- **Evolution Accuracy**: Performance after evolution
- **Improvement Rate**: Rate of accuracy gains over time

### Learning Efficiency
- **Data Efficiency**: How quickly accuracy improves with data
- **Evolution Frequency**: How often evolution occurs
- **Convergence Time**: Time to reach target accuracy

## 🚀 Deployment

### Production Build
```bash
# Build frontend
npm run build

# Start production server
npm start
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t apexverify-ai .
docker run -p 8000:8000 -p 3000:3000 apexverify-ai
```

## 🤝 Contributing

We welcome contributions to improve ApexVerify AI:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow TypeScript best practices
- Write tests for new features
- Update documentation
- Use conventional commit messages

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenEvolve Team**: For the excellent open-source implementation
- **AlphaEvolve Research**: For the foundational research
- **v0.dev**: For the beautiful UI components
- **Community Contributors**: For feedback and improvements

## 📞 Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/urban7733/ApexVerify/issues)
- **Documentation**: Check the comprehensive documentation
- **Community**: Join our community discussions

## 🔮 Roadmap

### v2.1 - Enhanced Learning
- Multi-modal learning (video + audio)
- Advanced evolution algorithms
- Distributed learning capabilities

### v2.2 - Creator Tools
- Trust badge marketplace
- Creator verification system
- Platform integrations

### v2.3 - Enterprise Features
- Multi-tenant architecture
- Advanced analytics dashboard
- API rate limiting and quotas

---

**ApexVerify AI with OpenEvolve** - Bringing self-learning AI to deepfake detection! 🚀

> "The last 10 years we paid to remove watermarks. The next decade, we're going to pay to have them." 