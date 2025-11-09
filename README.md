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
```
PROJECT_ID=your-GCP-project
EMBEDDING_MODEL=text-embedding-large-exp-03-07
LLM_MODEL=gemini-2.5-pro
PERSIST_DIRECTORY=./chroma_db
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
HTML_FILE_PATH=./index.html
GCS_BUCKET_NAME=your-gcs-buket-name
```
