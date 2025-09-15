"""
Modern Streamlit UI for PDF Q/A Agent.
Following SOLID principles with clean separation of concerns.
"""
import streamlit as st
import asyncio
from pathlib import Path
import tempfile
import os
from typing import Optional
import plotly.express as px
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Import our services
from main import PDFQAAgent
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class StreamlitUI:
    """Streamlit UI controller following Single Responsibility Principle."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=settings.app_title,
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .assistant-message {
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "pdf_agent" not in st.session_state:
            st.session_state.pdf_agent = None
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = []
        
        if "agent_initialized" not in st.session_state:
            st.session_state.agent_initialized = False
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">📚 PDF Q/A Agent</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with controls and information."""
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # Agent initialization
            if not st.session_state.agent_initialized:
                if st.button("🚀 Initialize Agent", type="primary"):
                    self.initialize_agent()
            else:
                st.success("✅ Agent Initialized")
            
            add_vertical_space(2)
            
            # Strategy selection
            st.header("⚙️ Advanced Settings")
            
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                ["hybrid", "recursive", "semantic", "contextual"],
                help="Choose how documents are split into chunks"
            )
            
            retrieval_strategy = st.selectbox(
                "Retrieval Strategy", 
                ["hybrid", "basic", "contextual"],
                help="Choose how relevant documents are retrieved"
            )
            
            if st.button("🔄 Update Strategies"):
                self.update_strategies(chunking_strategy, retrieval_strategy)
            
            add_vertical_space(2)
            
            # Knowledge base stats
            if st.session_state.agent_initialized:
                self.render_knowledge_base_stats()
            
            add_vertical_space(2)
            
            # Clear knowledge base
            if st.session_state.agent_initialized:
                if st.button("🗑️ Clear Knowledge Base", type="secondary"):
                    self.clear_knowledge_base()
    
    def render_file_upload(self):
        """Render file upload section."""
        st.header("📄 Upload PDF Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help=f"Maximum file size: {settings.max_file_size_mb}MB per file"
        )
        
        if uploaded_files and st.session_state.agent_initialized:
            if st.button("📤 Process Files", type="primary"):
                self.process_uploaded_files(uploaded_files)
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("📋 Processed Files")
            for file_info in st.session_state.processed_files:
                st.info(f"✅ {file_info['name']} - {file_info['chunks']} chunks")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        st.header("💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "Ask a question about your PDFs:",
            placeholder="What is the main topic discussed in the documents?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            ask_button = st.button("🤔 Ask", type="primary")
        
        with col2:
            if st.button("🧹 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and question and st.session_state.agent_initialized:
            self.handle_question(question)
    
    def initialize_agent(self):
        """Initialize the PDF Q/A agent."""
        try:
            with st.spinner("Initializing PDF Q/A Agent..."):
                st.session_state.pdf_agent = PDFQAAgent()
                st.session_state.agent_initialized = True
            st.success("Agent initialized successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            logger.error(f"Agent initialization failed: {e}")
    
    def update_strategies(self, chunking_strategy: str, retrieval_strategy: str):
        """Update agent strategies."""
        try:
            with st.spinner("Updating strategies..."):
                st.session_state.pdf_agent = PDFQAAgent(
                    chunking_strategy=chunking_strategy,
                    retrieval_strategy=retrieval_strategy
                )
            st.success("Strategies updated successfully!")
        except Exception as e:
            st.error(f"Failed to update strategies: {str(e)}")
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = Path(tmp_file.name)
                
                # Process the file
                success = asyncio.run(st.session_state.pdf_agent.process_pdf(tmp_path))
                
                if success:
                    # Get chunk count (simplified)
                    chunks_count = "Unknown"  # Could be enhanced to get actual count
                    
                    st.session_state.processed_files.append({
                        "name": uploaded_file.name,
                        "chunks": chunks_count
                    })
                    
                    st.success(f"✅ Processed {uploaded_file.name}")
                else:
                    st.error(f"❌ Failed to process {uploaded_file.name}")
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"File processing error: {e}")
            
            finally:
                progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        st.rerun()
    
    def handle_question(self, question: str):
        """Handle user question with streaming response."""
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Create placeholder for streaming response
        response_placeholder = st.empty()
        response_text = ""
        
        try:
            # Stream the response
            async def stream_response():
                nonlocal response_text
                async for chunk in st.session_state.pdf_agent.ask_question_stream(question):
                    response_text += chunk
                    response_placeholder.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {response_text}</div>', 
                                                unsafe_allow_html=True)
            
            # Run the async function
            asyncio.run(stream_response())
            
            # Add complete response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        # Rerun to update the chat display
        st.rerun()
    
    def render_knowledge_base_stats(self):
        """Render knowledge base statistics."""
        st.header("📊 Knowledge Base Stats")
        
        try:
            stats = st.session_state.pdf_agent.get_knowledge_base_stats()
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📚 Documents: {stats.get('document_count', 0)}</h4>
                <p>Collection: {stats.get('collection_name', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    
    def clear_knowledge_base(self):
        """Clear the knowledge base."""
        try:
            with st.spinner("Clearing knowledge base..."):
                success = st.session_state.pdf_agent.clear_knowledge_base()
            
            if success:
                st.session_state.processed_files = []
                st.success("Knowledge base cleared successfully!")
                st.rerun()
            else:
                st.error("Failed to clear knowledge base")
                
        except Exception as e:
            st.error(f"Error clearing knowledge base: {str(e)}")
    
    def run(self):
        """Main application runner."""
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_file_upload()
        
        with col2:
            if st.session_state.agent_initialized:
                self.render_chat_interface()
            else:
                st.info("👈 Please initialize the agent first using the sidebar")


def main():
    """Main application entry point."""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for required environment variables
        if not os.getenv("GROQ_API_KEY"):
            st.error("❌ GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
            st.stop()
        
        # Run the application
        app = StreamlitUI()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
