import os
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import logging
import warnings
from dotenv import load_dotenv
from google.cloud import storage
import shutil

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# -----------------------------
# GCS Upload Functions
# -----------------------------
def upload_directory_to_gcs(bucket_name: str, source_dir: str, destination_blob_prefix: str = ""):
    """Upload a directory to GCS recursively"""
    if not bucket_name:
        logger.warning("No GCS bucket name provided. Skipping upload.")
        return []
        
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    uploaded_files = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_dir)
            blob_path = os.path.join(destination_blob_prefix, relative_path).replace("\\", "/")
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            uploaded_files.append(blob_path)
            
            logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")
    
    logger.info(f"Successfully uploaded {len(uploaded_files)} files to GCS")
    return uploaded_files

def download_directory_from_gcs(bucket_name: str, source_blob_prefix: str, destination_dir: str):
    """Download a directory from GCS recursively"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_prefix)
    
    downloaded_files = []
    
    for blob in blobs:
        # Skip directories (they don't exist as separate blobs in GCS)
        if blob.name.endswith('/'):
            continue
            
        # Create local directory structure
        relative_path = os.path.relpath(blob.name, source_blob_prefix)
        local_path = os.path.join(destination_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        blob.download_to_filename(local_path)
        downloaded_files.append(local_path)
        
        logger.info(f"Downloaded gs://{bucket_name}/{blob.name} to {local_path}")
    
    return downloaded_files

# -----------------------------
# HTML Extraction
# -----------------------------
def extract_content_from_html(html_content: str) -> Dict[str, List[str]]:
    logger.info("Starting HTML content extraction")
    soup = BeautifulSoup(html_content, 'html.parser')
    content = {
        'overview': [],
        'current_role': [],
        'experience_details': [],
        'skills_detailed': [],
        'projects_detailed': [],
        'education_detailed': [],
        'certifications_detailed': [],
        'blog_posts_detailed': [],
        'contact_info': [],
        'key_metrics': [],
        'testimonials': [],
        'summary_activities': [],
        'tools_technologies': []
    }

    # Extract personal/professional overview from sidebar
    sidebar_info = soup.find('div', class_='sidebar-info')
    if sidebar_info:
        name_elem = sidebar_info.find('h1', class_='name')
        title_elem = sidebar_info.find('p', class_='title')
        if name_elem and title_elem:
            full_name = name_elem.get_text().strip()
            title = title_elem.get_text().strip()
            content['overview'].append(f"Full Name: {full_name}")
            content['overview'].append(f"Professional Title: {title}")
            content['current_role'].append(f"Mncedisi Lindani Mncwabe (also known as Lindani) is a {title}")

    # Extract about section
    about_section = soup.find('section', class_='about-text')
    if about_section:
        paragraphs = about_section.find_all('p')
        full_about = " ".join([p.get_text().strip() for p in paragraphs])
        content['overview'].append(f"Professional Background: {full_about}")

    # Extract summary activities
    summary_section = soup.find('section', class_='summary')
    if summary_section:
        summary_items = summary_section.find_all('li', class_='summary-item')
        for item in summary_items:
            title_elem = item.find('h4', class_='summary-item-title')
            text_elem = item.find('p', class_='summary-item-text')
            if title_elem and text_elem:
                activity = f"{title_elem.get_text().strip()}: {text_elem.get_text().strip()}"
                content['summary_activities'].append(activity)

    # Extract testimonials
    testimonials_section = soup.find('section', class_='testimonials')
    if testimonials_section:
        testimonial_items = testimonials_section.find_all('li', class_='testimonials-item')
        for item in testimonial_items:
            title_elem = item.find('h4', class_='testimonials-item-title')
            text_elem = item.find('div', class_='testimonials-text')
            if title_elem and text_elem:
                testimonial = f"From {title_elem.get_text().strip()}: {text_elem.get_text().strip()}"
                content['testimonials'].append(testimonial)

    # Extract key metrics
    metrics_section = soup.find('section', class_='metrics')
    if metrics_section:
        metric_items = metrics_section.find_all('li', class_='metrics-item')
        for item in metric_items:
            value_elem = item.find('h4', class_='metric-value')
            desc_elem = item.find('p', class_='metric-description')
            if value_elem and desc_elem:
                metric = f"{desc_elem.get_text().strip()}: {value_elem.get_text().strip()}"
                content['key_metrics'].append(metric)

    # Extract work experience
    experience_timeline = soup.find_all('section', class_='timeline')[1] if len(soup.find_all('section', class_='timeline')) > 1 else None
    if experience_timeline:
        exp_items = experience_timeline.find_all('li', class_='timeline-item')
        for item in exp_items:
            position_elem = item.find('h4', class_='timeline-item-title')
            date_elem = item.find('span')
            achievement_elems = item.find_all('li')  

            if position_elem:
                position = position_elem.get_text().strip()
                duration = date_elem.get_text().strip() if date_elem else "Duration not specified"
                exp_entry = f"Position: {position} | Duration: {duration}"

                if achievement_elems:
                    achievements = [ach.get_text().strip() for ach in achievement_elems]
                    exp_entry += f" | Achievements: {' | '.join(achievements)}"

                content['experience_details'].append(exp_entry)
                if 'Present' in duration:
                    content['current_role'].append(f"Currently {position} since {duration.split('—')[0].strip()}")

    # Extract education
    education_timeline = soup.find_all('section', class_='timeline')[0] if soup.find_all('section', class_='timeline') else None
    if education_timeline:
        edu_items = education_timeline.find_all('li', class_='timeline-item')
        for item in edu_items:
            institution_elem = item.find('h4', class_='timeline-item-title')
            date_elem = item.find('span')
            degree_elem = item.find('p', class_='timeline-text')

            if institution_elem and degree_elem:
                institution = institution_elem.get_text().strip()
                degree = degree_elem.get_text().strip()
                year = date_elem.get_text().strip() if date_elem else "Year not specified"
                edu_entry = f"Degree: {degree} | Institution: {institution} | Year: {year}"
                content['education_detailed'].append(edu_entry)

    # Extract skills
    skills_section = soup.find('section', class_='skills')
    if skills_section:
        skill_categories = skills_section.find_all('div', class_='skill-category')
        for category in skill_categories:
            category_name_elem = category.find('h4', class_='category-title')
            skill_items = category.find_all('li', class_='skill-item')

            if category_name_elem and skill_items:
                category_name = category_name_elem.get_text().strip()
                skills_list = [item.get_text().strip() for item in skill_items]
                skill_entry = f"Category: {category_name} | Skills: {', '.join(skills_list)}"
                content['skills_detailed'].append(skill_entry)

    # Extract tools & technologies
    tools_section = soup.find('section', class_='tools-technologies')
    if tools_section:
        tool_items = tools_section.find_all('div', class_='tool-item')
        tools_list = [item.find('p').get_text().strip() for item in tool_items if item.find('p')]
        if tools_list:
            content['tools_technologies'].append(f"Tools: {', '.join(tools_list)}")

    # Extract projects
    projects_section = soup.find('article', class_='projects')
    if projects_section:
        project_items = projects_section.find_all('li', class_='project-item')
        for item in project_items:
            title_elem = item.find('h3', class_='project-title')
            category_elem = item.find('p', class_='project-category')
            link = item.find('a').get('href', '') if item.find('a') else ''

            if title_elem:
                project_name = title_elem.get_text().strip()
                description = category_elem.get_text().strip() if category_elem else "No description"
                project_entry = f"Project: {project_name} | Description: {description}"
                if link:
                    project_entry += f" | Link: {link}"
                content['projects_detailed'].append(project_entry)

    # Extract certifications
    cert_section = soup.find('article', class_='certifications')
    if cert_section:
        cert_items = cert_section.find_all('li', class_='certifications-post-item')
        for item in cert_items:
            title_elem = item.find('h3', class_='certifications-item-title')
            text_elem = item.find('p', class_='certifications-text')
            category_elem = item.find('p', class_='certifications-category')
            date_elem = item.find('time')
            link = item.find('a').get('href', '') if item.find('a') else ''

            if title_elem:
                cert_name = title_elem.get_text().strip()
                description = text_elem.get_text().strip() if text_elem else "No description"
                cert_entry = f"Certification: {cert_name} | Description: {description}"
                if category_elem:
                    cert_entry += f" | Category: {category_elem.get_text().strip()}"
                if date_elem:
                    cert_entry += f" | Date: {date_elem.get_text().strip()}"
                if link:
                    cert_entry += f" | Link: {link}"
                content['certifications_detailed'].append(cert_entry)

    # Extract blog posts
    blog_section = soup.find('section', class_='blog-posts')
    if blog_section:
        blog_items = blog_section.find_all('li', class_='blog-post-item')
        for item in blog_items:
            title_elem = item.find('h3', class_='blog-item-title')
            text_elem = item.find('p', class_='blog-text')
            category_elem = item.find('p', class_='blog-category')
            date_elem = item.find('time')
            link = item.find('a').get('href', '') if item.find('a') else ''

            if title_elem:
                blog_title = title_elem.get_text().strip()
                description = text_elem.get_text().strip() if text_elem else "No description"
                blog_entry = f"Blog Post: {blog_title} | Description: {description}"
                if category_elem:
                    blog_entry += f" | Category: {category_elem.get_text().strip()}"
                if date_elem:
                    blog_entry += f" | Date: {date_elem.get_text().strip()}"
                if link:
                    blog_entry += f" | Link: {link}"
                content['blog_posts_detailed'].append(blog_entry)

    # Extract contact information
    contacts_list = soup.find('ul', class_='contacts-list')
    if contacts_list:
        contact_items = contacts_list.find_all('li', class_='contact-item')
        for item in contact_items:
            title_elem = item.find('p', class_='contact-title')
            link_elem = item.find('a', class_='contact-link')
            address_elem = item.find('address')
            if title_elem:
                contact_type = title_elem.get_text().strip()
                contact_value = link_elem.get_text().strip() if link_elem else (address_elem.get_text().strip() if address_elem else "")
                content['contact_info'].append(f"{contact_type}: {contact_value}")

    # Extract social links
    social_list = soup.find('ul', class_='social-list')
    if social_list:
        social_items = social_list.find_all('li', class_='social-item')
        social_links = []
        for item in social_items:
            link = item.find('a').get('href', '') if item.find('a') else ''
            if link:
                social_links.append(link)
        if social_links:
            content['contact_info'].append(f"Social Links: {', '.join(social_links)}")

    # Log extraction results
    total_items = sum(len(items) for items in content.values())
    logger.info(f"HTML extraction completed. Extracted {total_items} items across {len(content)} categories")
    
    return content

# -----------------------------
# Document Creation
# -----------------------------
def create_documents(content: Dict[str, List[str]]) -> List[Document]:
    logger.info("Creating document objects from extracted content")
    documents = []
    for category, items in content.items():
        if items:
            for item in items:
                if item.strip():
                    doc = Document(
                        page_content=item,
                        metadata={
                            'category': category,
                            'source': 'portfolio_website',
                            'person': 'Mncedisi Lindani Mncwabe',
                            'type': 'professional_info'
                        }
                    )
                    documents.append(doc)
            category_summary = f"Category: {category.replace('_', ' ').title()}\n\n" + "\n".join(items)
            summary_doc = Document(
                page_content=category_summary,
                metadata={
                    'category': f"{category}_summary",
                    'source': 'portfolio_website',
                    'person': 'Mncedisi Lindani Mncwabe',
                    'type': 'category_summary'
                }
            )
            documents.append(summary_doc)
    
    logger.info(f"Created {len(documents)} document objects")
    return documents

# -----------------------------
# Embeddings Model
# -----------------------------
def get_embeddings_model(model_name: str, project_id: str, location: str = "us-central1") -> VertexAIEmbeddings:
    logger.info(f"Initializing embeddings model: {model_name}")
    return VertexAIEmbeddings(
        model_name=model_name,
        project=project_id,
        location=location,
    )

# -----------------------------
# Vectorstore Setup & Persistence
# -----------------------------
def create_and_save_vectorstore(documents: List[Document], embeddings: VertexAIEmbeddings, persist_directory: str = "./chroma_db") -> Chroma:
    logger.info("Setting up vectorstore with document splitting")
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=[" | ", "\n\n", "\n", ". ", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)

    logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Create and persist vectorstore
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="portfolio_collection"
    )
    
    # Ensure persistence
    vectorstore.persist()
    
    logger.info(f"Vectorstore created and persisted to {persist_directory} with {len(split_docs)} embedded document chunks")
    return vectorstore

# -----------------------------
# Main Function for Embedding Creation
# -----------------------------
def main():
    logger.info("Starting portfolio embeddings creation pipeline")

    # Load variables from environment
    PROJECT_ID = os.getenv("PROJECT_ID", "fifth-medley-472712-t4")
    html_file_path = os.getenv("HTML_FILE_PATH", "./index.html")  # Default to index.html
    persist_directory = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-large-exp-03-07")
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME", "chromadb-portfolio-mncedisi") 

    # Read HTML content
    logger.info(f"Reading HTML file: {html_file_path}")
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Process content and create vectorstore
    content = extract_content_from_html(html_content)
    documents = create_documents(content)
    embeddings = get_embeddings_model(embedding_model, PROJECT_ID)
    vectorstore = create_and_save_vectorstore(documents, embeddings, persist_directory)
    
    # Test that the vectorstore can be loaded
    test_vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="portfolio_collection"
    )
    
    doc_count = test_vectorstore._collection.count()
    logger.info(f"Verified vectorstore persistence. Collection contains {doc_count} documents")
    
    # Upload to GCS if bucket name provided
    if gcs_bucket_name:
        logger.info(f"Uploading ChromaDB to GCS bucket: {gcs_bucket_name}")
        try:
            uploaded_files = upload_directory_to_gcs(gcs_bucket_name, persist_directory, "chroma_db")
            logger.info(f"Successfully uploaded {len(uploaded_files)} files to GCS")
            print(f"✅ ChromaDB uploaded to: gs://{gcs_bucket_name}/chroma_db/")
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {str(e)}")
            print(f"❌ GCS upload failed: {e}")
    else:
        logger.warning("GCS_BUCKET_NAME not set. ChromaDB will only be stored locally.")
        print("⚠️  GCS_BUCKET_NAME not set. ChromaDB stored locally only.")
    
    print(f"Embeddings successfully created and saved to: {persist_directory}")
    logger.info("Embeddings creation pipeline completed successfully")

if __name__ == "__main__":
    main()
