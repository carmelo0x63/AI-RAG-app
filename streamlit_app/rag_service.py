import os
import time
import logging
from typing import List, Dict, Any, Optional
import chromadb
import ollama
from chromadb.config import Settings
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, ollama_url: str = "http://ollama:11434", chroma_url: str = "http://chroma:8000"):
        self.ollama_url = ollama_url
        self.chroma_url = chroma_url
        self.ollama_client = None
        self.chroma_client = None
        self.collection = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Ollama and Chroma clients with retry logic."""
        max_retries = 10
        retry_delay = 5
        
        # Initialize Ollama client
        for attempt in range(max_retries):
            try:
                self.ollama_client = ollama.Client(host=self.ollama_url)
                # Test connection
                self.ollama_client.list()
                logger.info("Successfully connected to Ollama")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to connect to Ollama failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise Exception("Failed to connect to Ollama after maximum retries")
        
        # Initialize Chroma client
        for attempt in range(max_retries):
            try:
                self.chroma_client = chromadb.HttpClient(
                    host=self.chroma_url.replace("http://", "").split(":")[0],
                    port=int(self.chroma_url.split(":")[-1]),
                    settings=Settings(allow_reset=True)
                )
                # Test connection
                self.chroma_client.heartbeat()
                logger.info("Successfully connected to Chroma")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to connect to Chroma failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise Exception("Failed to connect to Chroma after maximum retries")
        
        # Initialize collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the document collection."""
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Document collection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            models = self.ollama_client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            logger.info(f"Pulling model: {model_name}")
            self.ollama_client.pull(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector database."""
        try:
            if not documents:
                return True
            
            # Prepare data for Chroma
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                doc_id = self._generate_doc_id(doc['content'], doc.get('metadata', {}))
                ids.append(doc_id)
                texts.append(doc['content'])
                metadatas.append(doc.get('metadata', {}))
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def _generate_doc_id(self, content: str, metadata: Dict) -> str:
        """Generate a unique ID for a document based on content and metadata."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        return f"{content_hash}_{metadata_hash}"
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents based on query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], model: str = "llama2") -> str:
        """Generate response using Ollama with retrieved context."""
        try:
            # Prepare context
            context = "\n\n".join([doc['content'] for doc in context_docs])
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                stream=False
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            return {
                'document_count': count,
                'collection_name': 'documents'
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'document_count': 0, 'collection_name': 'documents'}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name="documents")
            self._initialize_collection()
            logger.info("Successfully cleared document collection")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False