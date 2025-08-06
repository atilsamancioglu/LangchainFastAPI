"""
DocuChat - A Simple Document Q&A System
Tutorial: FastAPI + LangChain + OpenAI + ChromaDB
"""

import os
import tempfile
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
# Disable ChromaDB telemetry completely
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
import PyPDF2
import docx

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

# Global variables for our AI components
vectorstore = None
qa_chain = None
embeddings = None
llm = None

# Initialize ChromaDB client with settings to disable telemetry
chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False  # Ensure ephemeral mode
    )
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the application"""
    # Startup
    print("Starting DocuChat application...")
    
    # Initialize OpenAI components after environment variables are loaded
    global embeddings, llm
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Make sure you have a .env file with OPENAI_API_KEY=your-key")
        return

    try:
        # Set environment variable first
        os.environ["OPENAI_API_KEY"] = api_key

        # Simple, clean initialization
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o")

    except Exception as e:
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

    # Check ChromaDB connection
    try:
        # Try to list collections to test connection
        collections = chroma_client.list_collections()
        print(f"ChromaDB connected successfully. Found {len(collections)} collections.")
    except Exception as e:
        print(f"ChromaDB connection issue: {e}")

    yield

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DocuChat", 
    description="Simple Document Q&A System",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Serve the main page with file upload and chat interface"""
    try:
        # Read HTML content from template file
        html_file_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback error message if template file is missing
        error_html = """
        <html>
            <body style="font-family: Arial; text-align: center; margin-top: 100px;">
                <h1> Template Error</h1>
                <p>The template file 'templates/index.html' was not found.</p>
                <p>Please make sure the templates directory exists and contains index.html.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for Q&A"""
    
    print(f"Starting upload process for: {file.filename}")
    
    # Check if OpenAI components are initialized
    if embeddings is None:
        print("OpenAI embeddings not initialized")
        raise HTTPException(status_code=500, detail="OpenAI embeddings not initialized. Check your API key.")
    
    # Check if file type is supported
    supported_types = ['.pdf', '.txt', '.docx']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in supported_types:
        print(f"Unsupported file type: {file_extension}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {supported_types}")
    
    print(f"File type {file_extension} is supported")
    
    tmp_file_path = None
    
    try:
        print("Saving uploaded file temporarily...")
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        print(f"File saved to: {tmp_file_path}")
        
        print("Extracting text from file...")
        # Extract text from the file based on its type
        documents = extract_text_from_file(tmp_file_path, file.filename, file_extension)
        print(f"Extracted {len(documents)} document(s)")
        
        if not documents or not any(doc.page_content.strip() for doc in documents):
            print("No text content found in document")
            raise Exception("No text content could be extracted from the file")
        
        print("✂Splitting documents into chunks...")
        # Split documents into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise Exception("Could not create text chunks from the document")
        
        print("Adding chunks to vector store...")
        # Add chunks to vector store
        global vectorstore, qa_chain
        try:
            if vectorstore is None:
                print("Creating new vector store...")
                # Use the existing chroma_client to avoid instance conflicts
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name="documents",
                    embedding_function=embeddings
                )
                # Add documents to the empty vectorstore
                vectorstore.add_documents(chunks)
                print(" Created new vector store with existing client")
            else:
                vectorstore.add_documents(chunks)
                print("Added documents to existing vector store")
        except Exception as ve:
            print(f" Vector store error: {ve}")
            print(" Trying to reset and recreate vector store...")
            try:
                # Reset the client and try again
                chroma_client.reset()
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name="documents",
                    embedding_function=embeddings
                )
                vectorstore.add_documents(chunks)
                print("Created vector store after reset")
            except Exception as e2:
                print(f" Fallback also failed: {e2}")
                raise Exception(f"Vector store creation failed: {ve}. Fallback failed: {e2}")
        
        print("Creating Q&A chain...")
        # Create/update the Q&A chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        print("Q&A chain created successfully")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        print("Cleanup completed")
        return {"message": f"Successfully processed {file.filename}", "chunks": len(chunks)}
        
    except Exception as e:
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        # Clean up temporary file if it exists
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                print("Cleaned up temporary file after error")
            except Exception as cleanup_error:
                print(f"Could not clean up temp file: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def extract_text_from_file(file_path: str, filename: str, file_extension: str) -> List[Document]:
    """Extract text from different file types and return as LangChain documents"""
    
    print(f"Extracting text from {file_extension} file: {filename}")
    documents = []
    
    try:
        if file_extension == '.pdf':
            print("Processing PDF file...")
            # Handle PDF files
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"PDF has {len(pdf_reader.pages)} pages")
                text = ""
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    print(f" Extracted text from page {i+1}: {len(page_text)} characters")
                
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": filename}))
                    print(f"Created document with {len(text)} characters")
                else:
                    print(" No text extracted from PDF")
                
        elif file_extension == '.txt':
            print("Processing text file...")
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                print(f" Read {len(text)} characters from text file")
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": filename}))
                else:
                    print("Text file is empty")
                
        elif file_extension == '.docx':
            print("Processing Word document...")
            # Handle Word documents
            doc = docx.Document(file_path)
            print(f"Document has {len(doc.paragraphs)} paragraphs")
            text = ""
            for i, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text
                text += para_text + "\n"
                if para_text.strip():
                    print(f" Paragraph {i+1}: {len(para_text)} characters")
            
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": filename}))
                print(f" Created document with {len(text)} characters")
            else:
                print("No text extracted from Word document")
            
        print(f"Total documents created: {len(documents)}")
        return documents
            
    except Exception as e:
        import traceback
        raise Exception(f"Error reading {file_extension} file: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Ask a question about the uploaded documents"""
    
    # Check if OpenAI components are initialized
    if llm is None or embeddings is None:
        raise HTTPException(status_code=500, detail="OpenAI components not initialized. Check your API key.")
    
    # Check if we have documents loaded
    if qa_chain is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload documents first.")
    
    try:
        # Get answer from the Q&A chain
        result = qa_chain.invoke({"query": request.question})

        # Extract source information
        sources = []
        if "source_documents" in result:
            sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
            # Remove duplicates while preserving order
            sources = list(dict.fromkeys(sources))
        
        return ChatResponse(
            answer=result["result"],
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    # Check if template file exists
    template_exists = os.path.exists(os.path.join(os.path.dirname(__file__), "templates", "index.html"))
    
    return {
        "status": "healthy", 
        "message": "DocuChat is running!",
        "openai_embeddings_ready": embeddings is not None,
        "openai_llm_ready": llm is not None,
        "vectorstore_ready": vectorstore is not None,
        "qa_chain_ready": qa_chain is not None,
        "template_file_exists": template_exists
    }

if __name__ == "__main__":
    import uvicorn
    # Check if OpenAI API key is loaded from .env file
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found!")
    else:
        print(" OpenAI API key loaded successfully from .env file")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
