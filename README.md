# AI-Chatbot-for-Personal-Portfolio

I built and deployed this FastAPI backend that powers an intelligent AI chatbot for my personal portfolio. It is built using Google Vertex AI, LangChain, and ChromaDB to provide accurate, context-aware answers about my professional experience, skills, projects, and education.


### Features
- AI-Powered Q&A: Intelligent chatbot that answers questions about my career, education, skills, and projects
- RAG Architecture: Retrieval-Augmented Generation using ChromaDB vector store for accurate, context-aware responses
#### Google Vertex AI Integration:
- Embeddings: ```text-embedding-large-exp-03-07```
- LLM: ```gemini-2.5-pro```

#### Cloud-Native Deployment:
- Google Cloud Run for serverless hosting
- GCS-backed ChromaDB persistence
- CI/CD pipeline for automatic deployment via GitHub Actions
- Web Integration: Seamlessly integrated into portfolio website with chat interface



<img width="1333" height="822" alt="image" src="https://github.com/user-attachments/assets/9364982c-e2cf-45ef-b88e-fe831c515f1c" />


### Content Coverage
The AI chatbot can answer questions about:
- Professional experience such as current role, past positions, achievements
- Technical skills such as programming languages, frameworks, tools
- Detailed project descriptions, technologies used, outcomes
- Degrees, institutions, certifications
- Blog posts I've written
- Contact details
- Testimonials I received from previous or current collegues


### How to setup 

#### Prerequisites
- Python 3.11
- Google Cloud Project with Vertex AI enabled
- GCS Bucket for ChromaDB embeddings storage
- GCP Service account with appropriate permissions

1. Install dependencies: ``pip install -r requirements.txt``
2. Setup ENV Variables:
```
PROJECT_ID=YOUR-GCP-PROJECT
EMBEDDING_MODEL=YOUR-VERTEX-AI-EMBEDDINGS-MODEL
LLM_MODEL=YOUR-VERTEX-AI-LLM-MODEL
PERSIST_DIRECTORY=./chroma_db
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
HTML_FILE_PATH=./index.html # Your website's index.html file
GCS_BUCKET_NAME=YOUR-GCS-BUCKET-NAME
```
3. Create and store embeddings: ```python embeddings/create_embeddings.py```
  - HTML content extraction and parsing
  - Document chunking and processing
  - Precomputed vector embedding generation
  - GCS persistence for ChromaDB
  
4. Run the app : ```python model/app.py```
   - Similarity search against pre-existing vectors
   - Queries precomputed embeddings from ChromaDB (vector db)

### CI/CD Pipeline
The GitHub Actions workflow automatically:
- Builds Docker container on relevant file changes
- Pushes the Docker image to Google Artifact Registry
- Deploys to Cloud Run
- Manages environment variables securely

In your javascript file add the API endpoint from cloud run to query the FastAPI backend that processes questions against your precomputed embeddings and returns AI-generated responses about your portfolio:
```
// Send query to FastAPI backend - replace with your actual Cloud Run URL
fetch('https://portfolio-qa-41469873708.us-central1.run.app/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        question: messageText,
        format_type: 'html',
    }),
})
```


