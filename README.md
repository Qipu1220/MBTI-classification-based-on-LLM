# MBTI Personality Analysis Application

**CS117 - Computational Thinking Course Project**  
**Vietnam National University - University of Information Technology (VNU-UIT)**

## 📋 Project Overview

This project implements an intelligent MBTI (Myers-Briggs Type Indicator) personality analysis system using machine learning and natural language processing techniques. The application analyzes text responses to predict personality types based on the 16 MBTI personality categories.

## 🎯 Features

- **Semantic Analysis**: Uses advanced embedding models to understand text semantics
- **Style Analysis**: Analyzes writing style patterns for personality prediction
- **Vector-based Retrieval**: Efficient similarity search using vector databases
- **Response Deduplication**: Removes duplicate responses for better data quality
- **Interactive Web Interface**: Streamlit-based user-friendly interface
- **Real-time Analysis**: Instant personality type prediction from text input

## 🛠️ Technology Stack

- **Python 3.8+**
- **Streamlit** - Web application framework
- **PyTorch** - Deep learning framework
- **Sentence Transformers** - Semantic embedding models
- **NumPy & Pandas** - Data processing
- **FAISS/ChromaDB** - Vector similarity search
- **Google Gemini API** - Advanced text analysis

## 📁 Project Structure

```
Demo_code/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── src/                        # Core modules
│   ├── preprocessing.py        # Text preprocessing utilities
│   ├── embedding.py           # Semantic embedding generation
│   ├── style_embedding.py     # Writing style analysis
│   ├── retrieval.py           # Vector-based retrieval system
│   ├── deduplication.py       # Response deduplication
│   ├── prompt_builder.py      # Prompt construction for LLMs
│   └── pipeline.py            # Main analysis pipeline
├── app/                       # Application modules
│   └── generate/
│       └── gemini/            # Gemini API integration
├── mbti_dataset/              # MBTI training data
│   └── mbti_responses_800.json
└── image/                     # Project documentation
    └── Pipeline.drawio.pdf    # System architecture diagram
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Demo_code
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 💡 How It Works

1. **Text Input**: User provides text responses or personality-related content
2. **Preprocessing**: Text is cleaned, normalized, and prepared for analysis
3. **Embedding Generation**: 
   - Semantic embeddings capture meaning and context
   - Style embeddings analyze writing patterns
4. **Vector Retrieval**: Similar personality profiles are retrieved from the database
5. **Analysis Pipeline**: Multiple ML models analyze the text features
6. **Prediction**: MBTI personality type is predicted with confidence scores
7. **Results Display**: Interactive visualization of results and explanations

## 📊 MBTI Types Supported

The system can predict all 16 MBTI personality types:

| Analysts | Diplomats | Sentinels | Explorers |
|----------|-----------|-----------|-----------|
| INTJ - Architect | INFJ - Advocate | ISTJ - Logistician | ISTP - Virtuoso |
| INTP - Thinker | INFP - Mediator | ISFJ - Protector | ISFP - Adventurer |
| ENTJ - Commander | ENFJ - Protagonist | ESTJ - Executive | ESTP - Entrepreneur |
| ENTP - Debater | ENFP - Campaigner | ESFJ - Consul | ESFP - Entertainer |

## 🔬 Technical Approach

### Machine Learning Components
- **Semantic Embeddings**: all-MiniLM-L6-v2 model for text understanding
- **Style Analysis**: Custom features for writing style patterns
- **Vector Similarity**: FAISS for efficient similarity search
- **Ensemble Methods**: Multiple models for robust predictions

### Performance Optimizations
- GPU acceleration support
- Caching mechanisms for embeddings
- Batch processing for multiple inputs
- Efficient vector indexing

## 📈 Usage Examples

### Basic Text Analysis
```python
from src.pipeline import MBTIPipeline

# Initialize pipeline
pipeline = MBTIPipeline("mbti_dataset")
pipeline.initialize()

# Analyze text
text = "I love solving complex problems and thinking about future possibilities..."
result = pipeline.analyze(text)
print(f"Predicted MBTI Type: {result['mbti_type']}")
```

### Web Interface
1. Open the Streamlit application
2. Enter your text in the input area
3. Click "Analyze Personality"
4. View detailed results and explanations

## 📚 Course Context

This project was developed as part of the **CS117 - Computational Thinking** course at VNU - University of Information Technology. It demonstrates:

- **Computational Problem Solving**: Breaking down personality analysis into computational steps
- **Algorithm Design**: Implementing efficient text analysis algorithms
- **Data Structures**: Using appropriate data structures for text processing
- **System Design**: Creating a complete end-to-end application
- **Machine Learning Integration**: Applying ML techniques to real-world problems

## 🤝 Contributing

This is an academic project for CS117 course. For educational purposes:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request with detailed description

## 📄 License

This project is developed for educational purposes as part of CS117 coursework at VNU-UIT.

## 👥 Authors

**CS117 Students**  
Vietnam National University - University of Information Technology

## 🙏 Acknowledgments

- VNU-UIT CS117 Course Instructors
- Sentence Transformers library
- Streamlit framework
- Google Gemini API
- Open-source MBTI datasets

## 📞 Contact

For questions related to this CS117 project, please contact through the course communication channels.

---

**Vietnam National University - University of Information Technology**  
**CS117 - Computational Thinking Course Project**