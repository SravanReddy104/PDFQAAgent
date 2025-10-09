"""Modern Streamlit UI for PDF Q/A Agent.
Following SOLID principles with clean separation of concerns.
"""
import streamlit as st
import asyncio
from pathlib import Path
import tempfile
import os
import time
from typing import Optional, Dict, Any
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
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
        }
        .status-indicator {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            margin: 0.2rem 0;
            display: inline-block;
        }
        .status-processing {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .status-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .shadow-text {
            color: #6c757d;
            font-style: italic;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .loader {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .processing-steps {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #17a2b8;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "pdf_agent" not in st.session_state:
            st.session_state.pdf_agent = None
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = []
        
        if "agent_initialized" not in st.session_state:
            st.session_state.agent_initialized = False
            
        if "processing_status" not in st.session_state:
            st.session_state.processing_status = {}
            
        if "current_operation" not in st.session_state:
            st.session_state.current_operation = None
    
    def show_status_indicator(self, status: str, message: str):
        """Show a status indicator with appropriate styling."""
        status_class = f"status-{status}"
        st.markdown(f'<div class="status-indicator {status_class}">{message}</div>', 
                   unsafe_allow_html=True)
    
    def show_processing_steps(self, steps: list, current_step: int = -1):
        """Show processing steps with current step highlighted."""
        steps_html = '<div class="processing-steps"><h4>🔄 Processing Steps:</h4><ul>'
        
        for i, step in enumerate(steps):
            if i == current_step:
                steps_html += f'<li><strong>▶️ {step}</strong> <span class="loader"></span></li>'
            elif i < current_step:
                steps_html += f'<li>✅ {step}</li>'
            else:
                steps_html += f'<li>⏳ {step}</li>'
        
        steps_html += '</ul></div>'
        st.markdown(steps_html, unsafe_allow_html=True)
    
    def show_shadow_text(self, text: str, show_loader: bool = True):
        """Show shadow text indicating backend processes with optional loader."""
        if show_loader:
            loader_html = '<span class="loader"></span> '
        else:
            loader_html = ''
        
        st.markdown(f'<div class="shadow-text">{loader_html}{text}</div>', unsafe_allow_html=True)
    
    def show_loader_with_text(self, text: str):
        """Show a loader with text for active processes."""
        st.markdown(f'''
        <div class="shadow-text">
            <span class="loader"></span> {text}
        </div>
        ''', unsafe_allow_html=True)
    
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
            
            # Execution Mode Selection
            execution_mode = st.selectbox(
                "Execution Mode",
                ["chain", "graph"],
                index=0 if settings.execution_mode.lower() == "chain" else 1,
                help="Choose between traditional chain or LangGraph execution"
            )
            
            # Vector Store Selection
            vector_store_type = st.selectbox(
                "Vector Store Type",
                ["chroma", "pinecone"],
                index=0 if settings.vector_store_type.lower() == "chroma" else 1,
                help="Choose the vector database backend"
            )
            
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
            
            # LangGraph-specific options
            if execution_mode == "graph":
                st.subheader("🔗 LangGraph Options")
                
                checkpoint_backend = st.selectbox(
                    "Checkpoint Backend",
                    ["memory", "sqlite"],
                    help="Choose state persistence backend"
                )
                
                enable_human_in_loop = st.checkbox(
                    "Enable Human-in-the-Loop",
                    value=settings.enable_human_in_loop,
                    help="Enable human review and approval steps"
                )
                
                enable_summarization = st.checkbox(
                    "Enable Response Summarization",
                    value=settings.enable_summarization,
                    help="Automatically summarize long responses"
                )
                
                max_iterations = st.slider(
                    "Max Workflow Iterations",
                    min_value=1,
                    max_value=10,
                    value=settings.max_graph_iterations,
                    help="Maximum number of workflow iterations"
                )
            else:
                checkpoint_backend = "memory"
                enable_human_in_loop = False
                enable_summarization = False
                max_iterations = 3
            
            if st.button("🔄 Update Configuration"):
                self.update_configuration(
                    execution_mode, vector_store_type, chunking_strategy, 
                    retrieval_strategy, checkpoint_backend, enable_human_in_loop,
                    enable_summarization, max_iterations
                )
            
            add_vertical_space(2)
            
            # Knowledge base stats
            if st.session_state.agent_initialized:
                self.render_agent_status()
                self.render_knowledge_base_stats()
                self.render_web_search_status()
            
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
        """Render the modern chat interface using Streamlit's native components."""
        st.header("💬 Chat with your PDFs")
        
        # Show welcome message if no messages yet
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.markdown("""
                👋 **Welcome to PDF Q/A Agent!**
                
                I'm here to help you analyze and ask questions about your PDF documents. Here's how to get started:
                
                1. **Upload PDFs** using the file uploader on the left
                2. **Ask questions** about your documents using natural language
                3. **Get AI-powered answers** with references to your documents
                
                I can help you with:
                • Summarizing document content
                • Finding specific information
                • Comparing information across documents
                • Answering complex analytical questions
                
                Upload some PDFs and start asking questions! 🚀
                """)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show search information if available
                if "search_info" in message and message["search_info"]:
                    with st.expander("🌐 Search Details", expanded=False):
                        for search_detail in message["search_info"]:
                            st.markdown(f"• {search_detail}")
                
                # Show processing steps if available (legacy support)
                if "processing_steps" in message:
                    with st.expander("🔍 Processing Details", expanded=False):
                        for step in message["processing_steps"]:
                            st.markdown(f"- {step}")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDFs..."):
            if st.session_state.agent_initialized:
                self.handle_question(prompt)
            else:
                st.error("Please initialize the agent first using the sidebar.")
        
        # Clear chat button in sidebar would be better, but for now:
        if st.session_state.messages:
            if st.button("🧹 Clear Chat History", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()
    
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
    
    def update_configuration(self, execution_mode: str, vector_store_type: str, chunking_strategy: str, 
                           retrieval_strategy: str, checkpoint_backend: str = "memory", 
                           enable_human_in_loop: bool = False, enable_summarization: bool = True,
                           max_iterations: int = 3):
        """Update agent configuration including execution mode and LangGraph options."""
        try:
            with st.spinner("Updating configuration..."):
                # Update environment variables for this session
                import os
                os.environ["EXECUTION_MODE"] = execution_mode
                os.environ["VECTOR_STORE_TYPE"] = vector_store_type
                os.environ["CHECKPOINT_BACKEND"] = checkpoint_backend
                os.environ["ENABLE_HUMAN_IN_LOOP"] = str(enable_human_in_loop).lower()
                os.environ["ENABLE_SUMMARIZATION"] = str(enable_summarization).lower()
                os.environ["MAX_GRAPH_ITERATIONS"] = str(max_iterations)
                
                # Import and use the agent factory
                from services.agent_factory import AgentFactory, AgentWrapper
                
                # Create agent using factory
                agent = AgentFactory.create_agent(
                    execution_mode=execution_mode,
                    vector_store_type=vector_store_type,
                    chunking_strategy=chunking_strategy,
                    retrieval_strategy=retrieval_strategy,
                    checkpoint_backend=checkpoint_backend,
                    enable_human_in_loop=enable_human_in_loop,
                    enable_summarization=enable_summarization,
                    max_iterations=max_iterations
                )
                
                # Wrap agent for unified interface
                st.session_state.pdf_agent = AgentWrapper(agent, execution_mode)
                
                # Clear processed files as configuration may have changed
                st.session_state.processed_files = []
                
            # Success message with configuration details
            config_details = f"""
            **Configuration Updated Successfully!**
            - Execution Mode: {execution_mode.title()}
            - Vector Store: {vector_store_type.title()}
            - Chunking Strategy: {chunking_strategy.title()}
            - Retrieval Strategy: {retrieval_strategy.title()}
            """
            
            if execution_mode == "graph":
                config_details += f"""
            - Checkpoint Backend: {checkpoint_backend.title()}
            - Human-in-the-Loop: {'Enabled' if enable_human_in_loop else 'Disabled'}
            - Summarization: {'Enabled' if enable_summarization else 'Disabled'}
            - Max Iterations: {max_iterations}
                """
            
            st.success(config_details)
            st.info("ℹ️ Please re-upload your documents if you changed the vector store type.")
            
        except Exception as e:
            st.error(f"Failed to update configuration: {str(e)}")
            logger.error(f"Configuration update failed: {e}")
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files with modern progress indicators."""
        # Processing steps
        processing_steps = [
            "Saving uploaded files",
            "Extracting text content", 
            "Chunking documents",
            "Generating embeddings",
            "Storing in vector database"
        ]
        
        # Create containers for dynamic updates
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            file_progress = st.empty()
        
        with status_container:
            steps_placeholder = st.empty()
            shadow_text_placeholder = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Update file progress
                file_progress.info(f"📄 Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Show processing steps
                with steps_placeholder.container():
                    self.show_processing_steps(processing_steps, 0)
                
                # Step 1: Save file
                shadow_text_placeholder.empty()
                with shadow_text_placeholder.container():
                    self.show_loader_with_text("Saving file to temporary storage...")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = Path(tmp_file.name)
                
                # Step 2: Show processing status
                with steps_placeholder.container():
                    self.show_processing_steps(processing_steps, 1)
                with shadow_text_placeholder.container():
                    self.show_loader_with_text("Processing PDF document...")
                
                # Actually process the file
                success = asyncio.run(st.session_state.pdf_agent.process_pdf(tmp_path))
                
                if success:
                    # Show completion
                    with steps_placeholder.container():
                        self.show_processing_steps(processing_steps, len(processing_steps))
                    
                    with shadow_text_placeholder.container():
                        self.show_shadow_text("✅ File processing completed successfully!", show_loader=False)
                    
                    # Get chunk count (simplified)
                    chunks_count = "Unknown"  # Could be enhanced to get actual count
                    
                    st.session_state.processed_files.append({
                        "name": uploaded_file.name,
                        "chunks": chunks_count
                    })
                    
                    self.show_status_indicator("success", f"✅ Successfully processed {uploaded_file.name}")
                else:
                    self.show_status_indicator("error", f"❌ Failed to process {uploaded_file.name}")
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                self.show_status_indicator("error", f"❌ Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"File processing error: {e}")
            
            finally:
                progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Final status
        with shadow_text_placeholder.container():
            self.show_shadow_text("🎉 All files processed successfully!", show_loader=False)
        
        st.rerun()
    
    def handle_question(self, question: str):
        """Handle user question with real-time streaming response and search indicators."""
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        
        # Show assistant response with real-time processing
        with st.chat_message("assistant"):
            # Create containers for dynamic updates
            status_container = st.container()
            response_container = st.container()
            
            response_text = ""
            search_info = []
            
            try:
                # Show initial processing status
                with status_container:
                    status_placeholder = st.empty()
                    
                # Start with processing indicator
                with status_placeholder.container():
                    self.show_loader_with_text("Processing your question...")
                
                # Stream the actual response with real-time status updates
                with response_container:
                    response_placeholder = st.empty()
                    
                    async def stream_response_with_status():
                        nonlocal response_text, search_info
                        
                        # Check if agent supports streaming with callbacks
                        if hasattr(st.session_state.pdf_agent, 'ask_question_stream_with_callbacks'):
                            # Use enhanced streaming with search callbacks
                            async for chunk_data in st.session_state.pdf_agent.ask_question_stream_with_callbacks(question):
                                if isinstance(chunk_data, dict):
                                    # Handle status updates
                                    if chunk_data.get('type') == 'status':
                                        with status_placeholder.container():
                                            self.show_loader_with_text(chunk_data.get('message', ''))
                                    elif chunk_data.get('type') == 'search':
                                        search_info.append(chunk_data.get('message', ''))
                                        with status_placeholder.container():
                                            self.show_loader_with_text(f"🌐 {chunk_data.get('message', '')}")
                                    elif chunk_data.get('type') == 'content':
                                        response_text += chunk_data.get('content', '')
                                        response_placeholder.markdown(response_text)
                                else:
                                    # Regular text chunk
                                    response_text += str(chunk_data)
                                    response_placeholder.markdown(response_text)
                        else:
                            # Fallback to regular streaming with manual status updates
                            with status_placeholder.container():
                                self.show_loader_with_text("Searching knowledge base...")
                            
                            # Check if web search is enabled and show status
                            if hasattr(st.session_state.pdf_agent, 'web_search') and st.session_state.pdf_agent.web_search:
                                try:
                                    search_provider = st.session_state.pdf_agent.web_search.get_provider_info()
                                    if search_provider.get('available', False):
                                        with status_placeholder.container():
                                            provider_name = search_provider.get('provider', 'Web').title()
                                            self.show_loader_with_text(f"Searching {provider_name} for additional context...")
                                        search_info.append(f"Used {provider_name} search")
                                except Exception:
                                    pass
                            
                            # Show generation status
                            with status_placeholder.container():
                                self.show_loader_with_text("Generating AI response...")
                            
                            # Stream the response
                            async for chunk in st.session_state.pdf_agent.ask_question_stream(question):
                                response_text += chunk
                                response_placeholder.markdown(response_text)
                                
                                # Clear status once we start getting content
                                if len(response_text) > 50:  # After some content is received
                                    status_placeholder.empty()
                    
                    # Run the async function
                    asyncio.run(stream_response_with_status())
                
                # Clear any remaining status indicators
                status_placeholder.empty()
                
                # Add complete response to chat history
                message_data = {
                    "role": "assistant", 
                    "content": response_text
                }
                
                # Add search info if available
                if search_info:
                    message_data["search_info"] = search_info
                
                st.session_state.messages.append(message_data)
                
            except Exception as e:
                error_msg = f"❌ Error processing question: {str(e)}"
                with response_container:
                    st.error(error_msg)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
                logger.error(f"Question processing error: {e}")
        
        # Rerun to update the chat display
        st.rerun()
    
    def render_agent_status(self):
        """Render current agent status and execution mode."""
        st.header("🤖 Agent Status")
        
        try:
            if hasattr(st.session_state.pdf_agent, 'get_agent_info'):
                agent_info = st.session_state.pdf_agent.get_agent_info()
                
                # Execution mode indicator
                mode = agent_info.get('mode', 'unknown').title()
                agent_type = agent_info.get('type', 'Unknown')
                
                if mode.lower() == 'graph':
                    self.show_status_indicator("processing", f"🔗 {agent_type} Mode Active")
                else:
                    self.show_status_indicator("success", f"⚡ {agent_type} Mode Active")
                
                # Features
                features = agent_info.get('features', [])
                if features:
                    st.markdown("**Available Features:**")
                    for feature in features:
                        st.markdown(f"• {feature}")
                        
            else:
                # Fallback for older agent types
                execution_mode = getattr(settings, 'execution_mode', 'chain').title()
                self.show_status_indicator("success", f"⚡ {execution_mode} Mode Active")
                
        except Exception as e:
            st.error(f"Error loading agent status: {str(e)}")
    
    def render_knowledge_base_stats(self):
        """Render knowledge base statistics."""
        st.header("📊 Knowledge Base Stats")
        
        try:
            stats = st.session_state.pdf_agent.get_knowledge_base_stats()
            
            # Get current vector store type
            vector_store_type = getattr(settings, 'vector_store_type', 'Unknown').title()
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📚 Documents: {stats.get('document_count', 0)}</h4>
                <p>Collection: {stats.get('collection_name', 'Unknown')}</p>
                <p>Vector Store: <strong>{vector_store_type}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    
    def render_web_search_status(self):
        """Render web search status information."""
        st.header("🌐 Web Search Status")
        
        try:
            if hasattr(st.session_state.pdf_agent, 'web_search'):
                search_info = st.session_state.pdf_agent.web_search.get_provider_info()
                
                if search_info.get('available', False):
                    st.success(f"✅ {search_info.get('provider', 'Unknown').title()} Search Active")
                    st.info(f"Max Results: {search_info.get('max_results', 3)}")
                else:
                    if search_info.get('enabled', False):
                        st.warning("⚠️ Web Search Enabled but Provider Unavailable")
                    else:
                        st.info("ℹ️ Web Search Disabled")
            else:
                st.info("ℹ️ Web Search Not Available")
                
        except Exception as e:
            st.error(f"Error loading web search status: {str(e)}")
    
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
        
        # Check for required environment variables when using Groq provider
        if settings.llm_provider.lower() == "groq":
            if not os.getenv("GROQ_API_KEY"):
                st.error("❌ GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
                st.stop()
        
        # Check for Pinecone API key when using Pinecone vector store
        current_vector_store = os.getenv("VECTOR_STORE_TYPE", settings.vector_store_type).lower()
        if current_vector_store == "pinecone":
            if not os.getenv("PINECONE_API_KEY"):
                st.error("❌ PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")
                st.info("💡 You can switch to ChromaDB (local storage) in the sidebar if you don't have a Pinecone API key.")
                st.stop()
        
        # Check for Tavily API key when web search is enabled with Tavily
        if settings.enable_web_search and settings.web_search_provider.lower() == "tavily":
            if not os.getenv("TAVILY_API_KEY"):
                st.warning("⚠️ TAVILY_API_KEY not found. Web search will fall back to DuckDuckGo.")
        
        # Run the application
        app = StreamlitUI()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
