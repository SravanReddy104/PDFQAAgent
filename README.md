# PDF Q/A Agent

A modern, intelligent PDF question-answering system built with LangChain, LangGraph, and Streamlit.

## 🚀 Features

- **Modern Chat Interface**: ChatGPT-like experience with real-time streaming responses
- **Dual Execution Modes**: Choose between Chain (fast) or Graph (advanced) processing
- **Multiple Vector Stores**: Support for ChromaDB (local) and Pinecone (cloud)
- **Web Search Integration**: Enhanced answers with Tavily and DuckDuckGo search
- **Human-in-the-Loop**: Optional review and approval workflows
- **Real-time Processing**: Live status updates and search indicators

## 🛠️ Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

- [LangGraph Advantages](docs/LANGGRAPH_ADVANTAGES.md)
- [LangChain vs LangGraph Comparison](docs/LangChain_vs_LangGraph_Comparison.md)
- [Deprecation Fixes](docs/DEPRECATION_FIXES.md)
- [Tavily Integration](docs/TAVILY_DEPRECATION_FIX.md)

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_web_search.py

# Run with verbose output
pytest -v
```

## 🐛 Debug Tools

Debug utilities are available in the `debug/` folder for development purposes.

## 🏗️ Architecture

- **Chain Mode**: Traditional sequential processing for fast, simple Q&A
- **Graph Mode**: LangGraph workflow with state management, checkpoints, and middleware
- **Vector Stores**: ChromaDB for local storage, Pinecone for cloud deployment
- **Web Search**: Intelligent web search integration for enhanced context

## 🔧 Configuration

The application supports extensive configuration through environment variables and the Streamlit UI:

- Execution mode (chain/graph)
- Vector store type (chroma/pinecone)
- Chunking strategies (hybrid/recursive/semantic/contextual)
- Retrieval strategies (hybrid/basic/contextual)
- Web search providers (tavily/duckduckgo)

## 📄 License

This project is licensed under the MIT License.
