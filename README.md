# MBTI Personality Analysis Pipeline

A sophisticated text analysis system that predicts MBTI personality types using advanced NLP techniques and machine learning.

## 🌟 Features

- **Dual Embedding System**: Combines semantic and stylistic text analysis
- **Hybrid Retrieval**: Finds similar personality examples from training data
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Text Comparison**: Compare personality traits between different texts
- **Analysis History**: Track and review past analyses
- **Gemini AI Integration**: Advanced prompt generation and analysis

## 🏗️ Architecture

```
├── app.py                 # Streamlit web interface
├── src/                   # Core pipeline modules
│   ├── pipeline.py        # Main orchestration
│   ├── preprocessing.py   # Text cleaning and normalization
│   ├── embedding.py       # Semantic embeddings
│   ├── style_embedding.py # Style feature extraction
│   ├── retrieval.py       # Vector similarity search
│   ├── deduplication.py   # Response deduplication
│   └── prompt_builder.py  # Prompt engineering
├── app_package/generate/gemini/   # Gemini AI integration
│   ├── api_key_manager.py # API key management
│   ├── gemini.py          # Gemini API interface
│   └── reset_api_key.py   # Key rotation logic
├── mbti_dataset/          # MBTI training data
└── requirements.txt       # Dependencies
```

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Demo_code
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment** (Optional - for Gemini AI)
Create a `.env` file with your Gemini API keys:
```
GEMINI_API_KEY_1=your_api_key_here
GEMINI_API_KEY_2=your_second_key_here
# Add more keys as needed
```

## 🚀 Usage

### Web Interface (Recommended)

Start the Streamlit app:
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Features Available:

1. **Text Analysis**
   - Enter text to analyze for MBTI personality traits
   - Adjust similarity search parameters
   - View matching examples from training data
   - See generated analysis prompts

2. **Text Comparison**
   - Compare personality traits between two texts
   - Side-by-side analysis results
   - Similarity scoring

3. **Analysis History**
   - Review past analyses
   - Export results
   - Clear history

4. **Settings**
   - Pipeline status and configuration
   - API key management
   - System information

### Python API Usage

```python
from src.pipeline import MBTIPipeline

# Initialize pipeline
pipeline = MBTIPipeline()
pipeline.initialize()

# Analyze text
result = pipeline.analyze_text("I love working alone on complex problems...")

# Compare texts
comparison = pipeline.compare_texts(text1, text2)
```

## 🔬 How It Works

### 1. Text Preprocessing
- Unicode normalization
- Text cleaning and chunking
- Language detection

### 2. Dual Embedding Creation
- **Semantic**: Content meaning using sentence transformers
- **Style**: Writing patterns (sentence length, punctuation, etc.)

### 3. Similarity Search
- Vector database lookup
- Hybrid scoring (semantic + style)
- Ranked retrieval

### 4. Deduplication
- Remove similar responses
- Diverse example selection

### 5. Prompt Generation
- Structured analysis prompts
- Context-aware formatting

## 📊 Pipeline Components

### Semantic Embedder
- Uses sentence-transformers models
- 384-dimensional embeddings
- Captures text meaning and context

### Style Embedder  
- Extracts writing style features:
  - Average sentence length
  - Punctuation usage
  - Word complexity
  - Emotional markers

### Vector Retriever
- Efficient similarity search
- Configurable thresholds
- Metadata preservation

### Response Deduplicator
- Prevents redundant examples
- Maintains diversity
- Weighted scoring

## 🎯 MBTI Types Supported

- **Analysts**: NT (INTJ, INTP, ENTJ, ENTP)
- **Diplomats**: NF (INFJ, INFP, ENFJ, ENFP) 
- **Sentinels**: SJ (ISTJ, ISFJ, ESTJ, ESFJ)
- **Explorers**: SP (ISTP, ISFP, ESTP, ESFP)

## 🔧 Configuration

### Pipeline Parameters
- `embedding_dim`: Embedding dimensions (default: 384)
- `max_chunk_size`: Text chunk size (default: 512)
- `similarity_threshold`: Minimum similarity score (default: 0.1)

### Retrieval Settings
- `k`: Number of similar examples (default: 5)
- `semantic_weight`: Semantic vs style balance (default: 0.7)

## 📈 Performance

- **Initialization**: ~30-60 seconds (first run)
- **Analysis Speed**: ~1-3 seconds per text
- **Memory Usage**: ~500MB-1GB
- **Dataset Size**: 800 MBTI responses

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Structure
- **Modular Design**: Each component is independently testable
- **Type Hints**: Full type annotation support
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed operation logging

## 🔄 Data Flow

```
Input Text → Preprocessing → Embedding Creation → 
Similarity Search → Deduplication → Prompt Building → 
Analysis Results
```

## 📝 Example Usage

```python
# Basic analysis
text = "I prefer working independently on complex analytical problems."
result = pipeline.analyze_text(text)
print(f"Top match: {result['similar_responses'][0]['mbti_type']}")

# Advanced configuration
result = pipeline.analyze_text(
    text, 
    k=10,              # More examples
    semantic_weight=0.8 # More semantic focus
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Troubleshooting

### Common Issues

1. **Pipeline initialization fails**
   - Check all dependencies are installed
   - Ensure dataset file exists
   - Verify sufficient memory available

2. **Slow performance**
   - Reduce dataset size for testing
   - Adjust embedding dimensions
   - Use GPU acceleration if available

3. **Import errors**
   - Verify Python path configuration
   - Check all required packages installed
   - Ensure compatible versions

### Getting Help

- Check the [Issues](../../issues) page for known problems
- Review error logs in the Streamlit interface
- Verify system requirements are met

## 🔮 Future Enhancements

- [ ] GPU acceleration support
- [ ] Additional personality frameworks
- [ ] Real-time analysis API
- [ ] Batch processing capabilities
- [ ] Advanced visualization tools
- [ ] Model fine-tuning interface