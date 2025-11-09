from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import storage
import shutil
import uvicorn
import os
import logging
import markdown
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Google Cloud credentials
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

app = FastAPI(title="Portfolio Q&A API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GCS Download Function
# -----------------------------
def download_chromadb_from_gcs(bucket_name: str, source_blob_prefix: str, destination_dir: str):
    """Download ChromaDB from GCS on startup"""
    logger.info(f"Downloading ChromaDB from GCS bucket: {bucket_name}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_prefix)
    
    # Clear existing directory if it exists
    if os.path.exists(destination_dir):
        logger.info(f"Cleaning existing directory: {destination_dir}")
        shutil.rmtree(destination_dir)
    
    # Create fresh directory
    os.makedirs(destination_dir, exist_ok=True)
    
    downloaded_count = 0
    for blob in blobs:
        # Skip directories (they don't exist as separate blobs in GCS)
        if blob.name.endswith('/'):
            continue
            
        # Create local directory structure
        relative_path = os.path.relpath(blob.name, source_blob_prefix)
        local_path = os.path.join(destination_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the file
        blob.download_to_filename(local_path)
        downloaded_count += 1
        logger.info(f"Downloaded: {blob.name} -> {local_path}")
    
    if downloaded_count == 0:
        logger.warning(f"No files found in GCS bucket: {bucket_name}/{source_blob_prefix}")
        raise ValueError(f"No ChromaDB files found in GCS bucket: {bucket_name}")
    else:
        logger.info(f"Successfully downloaded {downloaded_count} ChromaDB files from GCS")
    
    return downloaded_count

# -----------------------------
# Model Getters
# -----------------------------
def get_embeddings_model(model_name: str, project_id: str, location: str = "us-central1") -> VertexAIEmbeddings:
    logger.info(f"Initializing embeddings model: {model_name}")
    return VertexAIEmbeddings(
        model_name=model_name,
        project=project_id,
        location=location,
    )

def get_llm_model(model_name: str, project_id: str, location: str = "us-central1") -> VertexAI:
    logger.info(f"Initializing LLM model: {model_name}")
    return VertexAI(
        model_name=model_name,
        project=project_id,
        location=location,
        temperature=0.1,
        max_output_tokens=2048,
        max_retries=3,
    )

# -----------------------------
# Enhanced Vectorstore Loading
# -----------------------------
def load_vectorstore(embeddings: VertexAIEmbeddings, persist_directory: str = "./chroma_db") -> Chroma:
    logger.info(f"Loading vectorstore from {persist_directory}")
    
    # Check if ChromaDB exists locally, if not download from GCS
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        logger.warning(f"ChromaDB not found locally at {persist_directory}. Checking GCS...")
        gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
        
        if gcs_bucket_name:
            try:
                downloaded_count = download_chromadb_from_gcs(
                    gcs_bucket_name, 
                    "chroma_db", 
                    persist_directory
                )
                logger.info("Successfully downloaded ChromaDB from GCS")
            except Exception as e:
                logger.error(f"Failed to download from GCS: {str(e)}")
                raise ValueError(f"Could not load ChromaDB from local storage or GCS: {str(e)}")
        else:
            raise ValueError("GCS_BUCKET_NAME not set and no local ChromaDB found")
    
    try:
        # Load the vectorstore
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="portfolio_collection"
        )
        
        doc_count = vectorstore._collection.count()
        logger.info(f"Successfully loaded vectorstore with {doc_count} documents")
        
        if doc_count == 0:
            raise ValueError("Vectorstore is empty. Please create embeddings first.")
            
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to load vectorstore: {str(e)}")
        raise

# -----------------------------
# Text Formatting Functions
# -----------------------------
def format_text_to_html(text: str) -> str:
    """Convert markdown-style text to HTML"""
    # Convert markdown to HTML
    html = markdown.markdown(text, extensions=['extra', 'nl2br'])
    return html

def format_text_simple(text: str) -> str:
    """Simple text formatting without HTML"""
    text = re.sub(r'### (.*)', r'\n\1:\n', text)
    text = re.sub(r'## (.*)', r'\n\1:\n', text)
    text = re.sub(r'# (.*)', r'\n\1:\n', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\* (.*)', r'• \1', text)
    text = re.sub(r'- (.*)', r'• \1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# -----------------------------
# QA Setup - Version agnostic approach
# -----------------------------
def setup_qa_chain(vectorstore: Chroma, llm: VertexAI):
    logger.info("Setting up QA chain")
    
    # Try multiple approaches to handle different LangChain versions
    try:
        # Approach 1: Try the newest LangChain syntax (0.1.0+)
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        
        logger.info("Using LangChain 0.1.0+ syntax")
        custom_prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant answering questions about Mncedisi Lindani Mncwabe (also known as Lindani), a Senior Data Scientist from South Africa. Use the provided context to give comprehensive, accurate answers.

Context: {context}

Question: {input}

Instructions:
- Answer based solely on the provided context.
- Structure your response with clear sections using headers (use ### for main sections)
- Use bullet points for lists (use • or - for bullet points)
- Include specific details: company names, dates, technologies, metrics, and achievements.
- When discussing experience, mention company names, positions, durations, and key accomplishments.
- For skills questions, list specific technologies and tools from the context using bullet points.
- For projects, include names, descriptions, GitHub links, and technologies used.
- For education, include degrees, institutions, and years.
- For blog posts, include titles, descriptions, and links.
- Include quantifiable metrics when available (percentages, improvements, etc.)
- If information is not in the context, clearly state this
- Format your response clearly with proper spacing between sections
- Use line breaks to separate different types of information

Answer:"""
        )
        
        combine_documents_chain = create_stuff_documents_chain(llm, custom_prompt)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 12})
        qa_chain = create_retrieval_chain(retriever, combine_documents_chain)
        
    except ImportError:
        try:
            # Approach 2: Try older LangChain syntax with RetrievalQA
            logger.info("Using RetrievalQA approach")
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate
            
            prompt_template = """You are an AI assistant answering questions about Mncedisi Lindani Mncwabe (also known as Lindani), a Senior Data Scientist from South Africa. Use the provided context to give comprehensive, accurate answers.

{context}

Question: {question}

Instructions:
- Answer based solely on the provided context.
- Structure your response with clear sections
- Use bullet points for lists
- Include specific details: company names, dates, technologies, metrics, and achievements.
- If information is not in the context, clearly state this

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 12}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
        except ImportError:
            # Approach 3: Manual RAG implementation
            logger.info("Using manual RAG implementation")
            qa_chain = ManualRAGChain(vectorstore, llm)
    
    return qa_chain

# Fallback manual RAG implementation
class ManualRAGChain:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 12})
    
    def invoke(self, input_dict):
        question = input_dict["input"]
        
        # Retrieve relevant documents - handle different method names across versions
        try:
            # Try newer method name first
            if hasattr(self.retriever, 'invoke'):
                docs = self.retriever.invoke(question)
            elif hasattr(self.retriever, 'get_relevant_documents'):
                # Older method name
                docs = self.retriever.get_relevant_documents(question)
            else:
                # Fallback: use vectorstore directly
                docs = self.vectorstore.similarity_search(question, k=12)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            # Final fallback
            docs = self.vectorstore.similarity_search(question, k=12)
        
        # Extract page content from documents
        context_parts = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                context_parts.append(doc.page_content)
            elif hasattr(doc, 'content'):
                context_parts.append(doc.content)
            else:
                context_parts.append(str(doc))
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are an AI assistant answering questions about Mncedisi Lindani Mncwabe (also known as Lindani), a Senior Data Scientist from South Africa. Use the provided context to give comprehensive, accurate answers.

Context: {context}

Question: {question}

Instructions:
- Answer based solely on the provided context.
- Structure your response with clear sections
- Use bullet points for lists
- Include specific details: company names, dates, technologies, metrics, and achievements.
- If information is not in the context, clearly state this

Answer:"""
        
        # Get answer from LLM
        try:
            answer = self.llm.invoke(prompt)
            # Handle different response formats
            if hasattr(answer, 'content'):
                answer_text = answer.content
            elif isinstance(answer, str):
                answer_text = answer
            else:
                answer_text = str(answer)
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            answer_text = f"Sorry, I encountered an error while processing your question: {str(e)}"
        
        return {"answer": answer_text}

# -----------------------------
# Query
# -----------------------------
def query_pipeline(qa_chain, question: str, format_type: str = "html") -> dict:
    logger.info(f"Processing query: '{question}'")
    
    try:
        # Handle different chain types
        if hasattr(qa_chain, 'invoke'):
            # New style chain
            result = qa_chain.invoke({"input": question})
            if "answer" in result:
                raw_answer = result["answer"]
            elif "result" in result:
                raw_answer = result["result"]
            else:
                raw_answer = str(result)
        elif hasattr(qa_chain, '__call__'):
            # Older style chain
            result = qa_chain({"query": question})
            if "result" in result:
                raw_answer = result["result"]
            elif "answer" in result:
                raw_answer = result["answer"]
            else:
                raw_answer = str(result)
        else:
            # Manual RAG chain or other type
            result = qa_chain.invoke({"input": question})
            raw_answer = result["answer"]
        
        # Format the answer as html or plain text
        if format_type == "html":
            formatted_answer = format_text_to_html(raw_answer)
        elif format_type == "plain":
            formatted_answer = format_text_simple(raw_answer)
        else:
            formatted_answer = raw_answer  
        
        logger.info("Query processed successfully")
        return {
            "answer": formatted_answer,
            "raw_answer": raw_answer,  
            "format_type": format_type
        }
        
    except Exception as e:
        logger.error(f"Error in query pipeline: {str(e)}")
        raise

# -----------------------------
# FastAPI Setup
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    format_type: str = "html" 

# Global variables for models and vectorstore
PROJECT_ID = os.getenv("PROJECT_ID", "fifth-medley-472712-t4")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-large-exp-03-07")
llm_model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
persist_directory = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
gcs_bucket_name = os.getenv("GCS_BUCKET_NAME", "chromadb-portfolio-mncedisi")

# Initialize models and vectorstore
embeddings = None
llm = None
vectorstore = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    global embeddings, llm, vectorstore, qa_chain
    
    logger.info("Initializing models and vectorstore")
    
    try:
        # Check if required environment variables are set
        if not PROJECT_ID:
            raise ValueError("PROJECT_ID environment variable is not set")
        
        # Initialize models
        embeddings = get_embeddings_model(embedding_model, PROJECT_ID)
        llm = get_llm_model(llm_model, PROJECT_ID)
        
        # Load vectorstore (this will automatically download from GCS if needed)
        vectorstore = load_vectorstore(embeddings, persist_directory)
        qa_chain = setup_qa_chain(vectorstore, llm)
        
        logger.info("Initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

@app.post("/query")
async def query(request: QueryRequest):
    try:
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="QA chain not initialized")
            
        result = query_pipeline(qa_chain, request.question, request.format_type)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vectorstore_loaded": vectorstore is not None,
        "qa_chain_ready": qa_chain is not None,
        "gcs_bucket": os.getenv("GCS_BUCKET_NAME", "not set")
    }

@app.get("/")
async def root():
    return {
        "message": "Portfolio Q&A API is running",
        "version": "1.0",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
