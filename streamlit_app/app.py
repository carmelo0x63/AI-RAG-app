import streamlit as st
import os
import logging
from typing import List, Dict, Any
import time
from rag_service import RAGService
from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.success-message {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.error-message {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #1f77b4;
}

.user-message {
    background-color: #e3f2fd;
    border-left-color: #2196f3;
}

.bot-message {
    background-color: #f5f5f5;
    border-left-color: #4caf50;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_service' not in st.session_state:
        st.session_state.rag_service = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'available_models' not in st.session_state:
        st.session_state.available_models = []

def initialize_rag_service():
    """Initialize the RAG service."""
    if st.session_state.rag_service is None:
        try:
            with st.spinner("Initializing RAG service..."):
                ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
                chroma_url = os.getenv("CHROMA_URL", "http://chroma:8000")
                st.session_state.rag_service = RAGService(ollama_url, chroma_url)
                st.session_state.available_models = st.session_state.rag_service.get_available_models()
            st.success("✅ RAG service initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG service: {str(e)}")
            st.info("Please ensure Ollama and Chroma services are running.")
            return False
    return True

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Application</h1>', unsafe_allow_html=True)
    st.markdown("**Retrieval-Augmented Generation with Ollama, Chroma, and Streamlit**")
    
    # Initialize RAG service
    if not initialize_rag_service():
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Configuration")
        
        # Model selection
        st.subheader("🧠 Model Management")
        
        # Available models
        if st.session_state.available_models:
            selected_model = st.selectbox(
                "Select Model:",
                st.session_state.available_models,
                key="selected_model"
            )
        else:
            st.warning("No models available. Please pull a model first.")
            selected_model = None
        
        # Model pulling
        st.subheader("📥 Pull New Model")
        model_to_pull = st.text_input(
            "Model name (e.g., llama2, mistral, codellama):",
            placeholder="llama2"
        )
        
        if st.button("Pull Model", type="primary"):
            if model_to_pull:
                with st.spinner(f"Pulling {model_to_pull}... This may take several minutes."):
                    success = st.session_state.rag_service.pull_model(model_to_pull)
                    if success:
                        st.success(f"✅ Successfully pulled {model_to_pull}")
                        st.session_state.available_models = st.session_state.rag_service.get_available_models()
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to pull {model_to_pull}")
            else:
                st.warning("Please enter a model name")
        
        # Collection stats
        st.subheader("📊 Collection Stats")
        if st.session_state.rag_service:
            stats = st.session_state.rag_service.get_collection_stats()
            st.markdown(f"""
            <div class="stat-card">
                <h3>{stats['document_count']}</h3>
                <p>Documents in Collection</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear collection
        if st.button("🗑️ Clear Collection", type="secondary"):
            if st.session_state.rag_service.clear_collection():
                st.success("✅ Collection cleared!")
                st.session_state.documents_loaded = False
                st.rerun()
            else:
                st.error("❌ Failed to clear collection")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "💬 Chat", "🔍 Search"])
    
    with tab1:
        st.header("📄 Document Upload & Processing")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size (tokens)", 500, 2000, 1000, 100)
        with col2:
            chunk_overlap = st.slider("Chunk Overlap (tokens)", 50, 500, 200, 50)
        
        if uploaded_files:
            st.subheader("📁 Uploaded Files")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
            
            if st.button("🔄 Process Documents", type="primary"):
                process_documents(uploaded_files, chunk_size, chunk_overlap)
    
    with tab2:
        st.header("💬 Chat with Your Documents")
        
        if not st.session_state.documents_loaded:
            st.info("Please upload and process some documents first.")
        elif not selected_model:
            st.warning("Please select a model from the sidebar.")
        else:
            # Chat interface
            chat_interface(selected_model)
    
    with tab3:
        st.header("🔍 Document Search")
        
        if not st.session_state.documents_loaded:
            st.info("Please upload and process some documents first.")
        else:
            search_interface()

def process_documents(uploaded_files, chunk_size: int, chunk_overlap: int):
    """Process uploaded documents."""
    try:
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_documents = []
        
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                try:
                    # Reset file pointer
                    file.seek(0)
                    documents = processor.process_uploaded_file(file)
                    all_documents.extend(documents)
                    st.success(f"✅ Processed {file.name}: {len(documents)} chunks")
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        if all_documents:
            with st.spinner("Adding documents to vector database..."):
                success = st.session_state.rag_service.add_documents(all_documents)
                if success:
                    st.success(f"✅ Successfully added {len(all_documents)} document chunks to the database!")
                    st.session_state.documents_loaded = True
                else:
                    st.error("❌ Failed to add documents to the database")
        else:
            st.warning("No documents were successfully processed")
            
    except Exception as e:
        st.error(f"❌ Error during document processing: {str(e)}")

def chat_interface(selected_model: str):
    """Chat interface for RAG queries."""
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Assistant:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.spinner("Searching documents and generating response..."):
            try:
                # Search for relevant documents
                relevant_docs = st.session_state.rag_service.search_documents(query, n_results=5)
                
                if relevant_docs:
                    # Generate response
                    response = st.session_state.rag_service.generate_response(
                        query, relevant_docs, selected_model
                    )
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    st.rerun()
                else:
                    st.warning("No relevant documents found for your query.")
                    
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def search_interface():
    """Document search interface."""
    st.subheader("🔍 Search Your Documents")
    
    search_query = st.text_input("Enter your search query:", placeholder="What are you looking for?")
    n_results = st.slider("Number of results:", 1, 10, 5)
    
    if st.button("Search", type="primary") and search_query:
        with st.spinner("Searching documents..."):
            try:
                results = st.session_state.rag_service.search_documents(search_query, n_results)
                
                if results:
                    st.subheader(f"📋 Found {len(results)} relevant documents:")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} - {result['metadata'].get('filename', 'Unknown')}"):
                            st.write("**Content:**")
                            st.write(result['content'])
                            
                            st.write("**Metadata:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**File:** {result['metadata'].get('filename', 'N/A')}")
                            with col2:
                                st.write(f"**Type:** {result['metadata'].get('file_type', 'N/A')}")
                            with col3:
                                st.write(f"**Chunk:** {result['metadata'].get('chunk_index', 'N/A')}")
                            
                            if result.get('distance') is not None:
                                st.write(f"**Similarity Score:** {1 - result['distance']:.3f}")
                else:
                    st.info("No relevant documents found for your search query.")
                    
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")

if __name__ == "__main__":
    main()
        