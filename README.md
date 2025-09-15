# PDF Q/A Agent

A sophisticated PDF Question & Answer agent built with modern AI technologies, following SOLID principles for maintainable and extensible code.

## 🚀 Features

- **Advanced PDF Processing**: Multiple chunking strategies (recursive, semantic, contextual, hybrid)
- **Intelligent Retrieval**: Hybrid search combining similarity and keyword matching
- **Streaming Responses**: Real-time answer generation using Groq LLM
- **Modern UI**: Beautiful Streamlit interface with responsive design
- **Vector Database**: ChromaDB for efficient similarity search
- **Configurable**: Easy to customize strategies and parameters

## 🏗️ Architecture

The application follows SOLID principles with clean separation of concerns:

```
pdf_qa_agent/
├── config/          # Configuration management
├── core/            # Interface definitions
├── services/        # Business logic implementations
├── utils/           # Utility functions
├── main.py          # Application orchestrator
└── streamlit_app.py # User interface
```

## 📋 Prerequisites

- Python 3.8+
- Groq API key (get from [console.groq.com](https://console.groq.com))

## 🛠️ Installation

1. **Clone and navigate to the project:**
   ```bash
   cd pdf_qa_agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Download required NLTK data (optional for advanced chunking):**
   ```python
   import nltk
   nltk.download('punkt')
   ```

## 🚀 Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Using the Core API

```python
from main import PDFQAAgent
from pathlib import Path
import asyncio

# Initialize agent
agent = PDFQAAgent(
    chunking_strategy="hybrid",
    retrieval_strategy="hybrid"
)

# Process a PDF
await agent.process_pdf(Path("document.pdf"))

# Ask questions
response = await agent.ask_question("What is the main topic?")
print(response)

# Stream responses
async for chunk in agent.ask_question_stream("Explain the methodology"):
    print(chunk, end="")
```

## 🎯 Chunking Strategies

### 1. Recursive Chunking
- **Best for**: General documents
- **Method**: Splits on paragraphs, sentences, then words
- **Pros**: Fast, consistent chunk sizes
- **Cons**: May break semantic coherence

### 2. Semantic Chunking
- **Best for**: Long documents with topic shifts
- **Method**: Uses embeddings to detect topic boundaries
- **Pros**: Preserves semantic coherence
- **Cons**: Slower, variable chunk sizes

### 3. Contextual Chunking
- **Best for**: Complex documents requiring context
- **Method**: Adds document summary to each chunk
- **Pros**: Rich context preservation
- **Cons**: Larger chunks, more tokens

### 4. Hybrid Chunking (Recommended)
- **Best for**: Most use cases
- **Method**: Automatically selects strategy based on document
- **Pros**: Balanced performance and quality
- **Cons**: Slightly more complex

## 🔍 Retrieval Strategies

### 1. Basic Retrieval
- Simple similarity search
- Fast and straightforward

### 2. Hybrid Retrieval (Recommended)
- Combines similarity and keyword matching
- Weighted scoring for better relevance

### 3. Contextual Retrieval
- Query expansion with related terms
- Advanced ranking algorithms

## ⚙️ Configuration

Key settings in `config/settings.py`:

```python
# Model Configuration
groq_model = "llama3-8b-8192"
embedding_model = "all-MiniLM-L6-v2"

# Chunking Configuration
chunk_size = 1000
chunk_overlap = 200

# Retrieval Configuration
retrieval_k = 5
similarity_threshold = 0.7
```

## 🎨 UI Features

- **File Upload**: Drag & drop PDF files
- **Strategy Selection**: Choose chunking and retrieval methods
- **Real-time Chat**: Streaming responses
- **Knowledge Base Stats**: Document count and collection info
- **Chat History**: Persistent conversation history
- **Responsive Design**: Works on desktop and mobile

## 🔧 Advanced Usage

### Custom Chunking Strategy

```python
from core.interfaces import ChunkingStrategy

class CustomChunkingStrategy(ChunkingStrategy):
    def chunk_text(self, text: str, metadata: dict) -> List[dict]:
        # Your custom chunking logic
        pass
```

### Custom Retrieval Strategy

```python
from core.interfaces import RetrieverStrategy

class CustomRetrieverStrategy(RetrieverStrategy):
    def retrieve(self, query: str, vector_store) -> List[dict]:
        # Your custom retrieval logic
        pass
```

## 📊 Performance Tips

1. **For large documents**: Use semantic chunking
2. **For speed**: Use recursive chunking with basic retrieval
3. **For accuracy**: Use hybrid strategies
4. **Memory optimization**: Adjust chunk_size based on your needs

## 🐛 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Ensure `.env` file exists with valid API key

2. **ChromaDB errors**
   - Delete `chroma_db` folder and restart

3. **Memory issues**
   - Reduce `chunk_size` in settings
   - Process fewer documents at once

4. **Slow performance**
   - Use "basic" retrieval strategy
   - Reduce `retrieval_k` value

## 🤝 Contributing

1. Follow SOLID principles
2. Add comprehensive logging
3. Include type hints
4. Write unit tests
5. Update documentation

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) for document processing
- [Groq](https://groq.com) for fast LLM inference
- [ChromaDB](https://www.trychroma.com) for vector storage
- [Streamlit](https://streamlit.io) for the UI framework
