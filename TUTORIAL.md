# 🚀 Building DocuChat: A Complete FastAPI + LangChain Tutorial

## Welcome! 👋

In this 6-hour tutorial, you'll build **DocuChat** - a modern web application that lets users upload documents and ask questions about them using AI. By the end, you'll have a fully functional document Q&A system!

### What You'll Learn
- ⚡ **FastAPI**: Building modern APIs
- 🤖 **LangChain**: Working with AI and documents
- 🧠 **OpenAI**: Using GPT-4 for intelligent responses
- 💾 **Vector Databases**: Storing and searching document embeddings
- 🎨 **Web Development**: Creating beautiful user interfaces

### Prerequisites
- Basic Python knowledge (variables, functions, loops)
- Understanding of what APIs are
- Familiarity with basic web concepts (HTML, HTTP)

---

## 📋 Table of Contents

**Part 1: Project Setup (30 minutes)**
- [Step 1: Environment Setup](#step-1-environment-setup)
- [Step 2: Installing Dependencies](#step-2-installing-dependencies)
- [Step 3: Basic Project Structure](#step-3-basic-project-structure)

**Part 2: Basic FastAPI App (45 minutes)**
- [Step 4: Creating Your First FastAPI App](#step-4-creating-your-first-fastapi-app)
- [Step 5: Adding a Simple Web Interface](#step-5-adding-a-simple-web-interface)
- [Step 6: Testing Your Basic App](#step-6-testing-your-basic-app)

**Part 3: OpenAI Integration (60 minutes)**
- [Step 7: Setting Up OpenAI](#step-7-setting-up-openai)
- [Step 8: Creating the Chat Endpoint](#step-8-creating-the-chat-endpoint)
- [Step 9: Testing AI Responses](#step-9-testing-ai-responses)

**Part 4: Document Processing (90 minutes)**
- [Step 10: File Upload Functionality](#step-10-file-upload-functionality)
- [Step 11: Text Extraction from Documents](#step-11-text-extraction-from-documents)
- [Step 12: Document Chunking](#step-12-document-chunking)

**Part 5: Vector Database & RAG (90 minutes)**
- [Step 13: Setting Up ChromaDB](#step-13-setting-up-chromadb)
- [Step 14: Creating Embeddings](#step-14-creating-embeddings)
- [Step 15: Building the RAG System](#step-15-building-the-rag-system)
- [Step 15.5: Prompt Engineering for Better Results](#step-155-prompt-engineering-for-better-results)

**Part 6: Advanced Features & Polish (75 minutes)**
- [Step 16: Enhanced Error Handling](#step-16-enhanced-error-handling)
- [Step 17: Beautiful Web Interface](#step-17-beautiful-web-interface)
- [Step 18: Final Testing & Deployment](#step-18-final-testing--deployment)

---

## Part 1: Project Setup (30 minutes)

### Step 1: Environment Setup

First, let's create a clean workspace for our project.

#### 1.1 Create Project Directory
```bash
# Create a new folder for your project
mkdir DocuChat
cd DocuChat
```

#### 1.2 Create Virtual Environment
```bash
# Create a virtual environment (isolated Python environment)
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate
```

**💡 Why virtual environments?** They keep your project dependencies separate from other Python projects, preventing conflicts.

#### 1.3 Create Basic Files
```bash
# Create these empty files
touch main.py
touch requirements.txt
touch .env
touch README.md
```

### Step 2: Installing Dependencies

#### 2.1 Understanding Our Dependencies

Before installing, let's understand what each package does:

- **fastapi**: Web framework for building APIs quickly
- **uvicorn**: Web server to run our FastAPI app
- **langchain**: Framework for building AI applications
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-chroma**: Vector database integration
- **openai**: Direct access to OpenAI's API
- **python-dotenv**: Load environment variables from .env files

#### 2.2 Create requirements.txt
Open `requirements.txt` and add:

```txt
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.2.14
langchain-openai==0.1.23
langchain-community==0.2.12
langchain-chroma==0.1.4
chromadb==0.4.15
python-multipart==0.0.6
pypdf2==3.0.1
python-docx==1.1.0
pydantic==2.8.2
openai>=1.55.3
python-dotenv==1.0.1
```

#### 2.3 Install Dependencies
```bash
pip install -r requirements.txt
```

**⏳ This might take a few minutes!** Perfect time for a coffee break ☕

### Step 3: Basic Project Structure

#### 3.1 Understanding the Architecture

Our app will have this structure:
```
DocuChat/
├── main.py              # Main application code
├── templates/           # HTML templates
│   └── index.html      # Web interface
├── requirements.txt     # Dependencies
├── .env                # Environment variables
└── README.md           # Documentation
```

#### 3.2 Create Templates Directory
```bash
mkdir templates
```

---

## Part 2: Basic FastAPI App (45 minutes)

### Step 4: Creating Your First FastAPI App

Let's start with the absolute basics and build up gradually.

#### 4.1 Create Minimal FastAPI App

Open `main.py` and start with this simple code:

```python
# Import the FastAPI framework
from fastapi import FastAPI

# Create a FastAPI application instance
app = FastAPI(
    title="DocuChat",
    description="A Simple Document Q&A System"
)

# Define a simple route
@app.get("/")
async def root():
    """This function handles requests to the home page"""
    return {"message": "Hello! DocuChat is running!"}

# Define a health check endpoint
@app.get("/health")
async def health_check():
    """Check if our app is running properly"""
    return {"status": "healthy", "message": "DocuChat is working!"}

# Run the app (only when this file is run directly)
if __name__ == "__main__":
    import uvicorn
    print("Starting DocuChat...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 4.2 Test Your First App

```bash
python main.py
```

Open your browser and go to:
- `http://localhost:8000` - You should see a JSON message
- `http://localhost:8000/health` - Health check
- `http://localhost:8000/docs` - **Cool!** FastAPI automatically creates API documentation

**🎉 Congratulations!** You just created your first API!

### Step 5: Adding a Simple Web Interface

Now let's add a basic HTML interface so users can interact with our app.

#### 5.1 Update main.py with HTML Response

Add these imports at the top of `main.py`:

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse  # New import for HTML
import os  # New import for file operations
```

Replace your root endpoint with:

```python
@app.get("/")
async def root():
    """Serve a simple HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DocuChat - Document Q&A</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background-color: #f5f5f5;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #4CAF50; text-align: center; }
            .status { 
                padding: 15px; 
                background: #e8f5e8; 
                border-radius: 5px;
                margin: 20px 0;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 DocuChat</h1>
            <div class="status">
                <h3>✅ System Status: Running</h3>
                <p>Your DocuChat application is up and running!</p>
                <p>We'll add document upload and chat features soon.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
```

#### 5.2 Test Your Web Interface

Restart your app and visit `http://localhost:8000`. You should now see a nice web page!

**🎨 Much better!** Now users see a friendly interface instead of raw JSON.

### Step 6: Testing Your Basic App

Let's make sure everything works properly before moving forward.

#### 6.1 Add Error Handling

Update your `main.py` to handle errors gracefully:

```python
@app.get("/")
async def root():
    """Serve a simple HTML page"""
    try:
        # We'll read from a file later, for now keep it simple
        html_content = """
        <!-- Your HTML content here -->
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        # If something goes wrong, show a simple error
        error_html = f"""
        <html>
            <body style="text-align: center; margin-top: 100px;">
                <h1>😔 Oops!</h1>
                <p>Something went wrong: {str(e)}</p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)
```

**💡 Why error handling?** Real applications need to handle unexpected situations gracefully.

---

## Part 3: OpenAI Integration (60 minutes)

### Step 7: Setting Up OpenAI

Now we'll add AI capabilities to our app!

#### 7.1 Get Your OpenAI API Key

1. Go to [OpenAI's website](https://platform.openai.com)
2. Create an account or sign in
3. Go to API Keys section
4. Create a new API key
5. Copy it (you won't see it again!)

#### 7.2 Configure Environment Variables

Open your `.env` file and add:

```env
OPENAI_API_KEY=your-api-key-here
```

**🔒 Important:** Never commit your API key to version control!

#### 7.3 Update main.py with OpenAI Setup

Add these imports at the top:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import OpenAI components
from langchain_openai import ChatOpenAI
```

Add this global variable and initialization:

```python
# Global variables for our AI components
llm = None  # This will hold our ChatGPT model

# Initialize FastAPI app
app = FastAPI(
    title="DocuChat",
    description="A Simple Document Q&A System"
)

# Initialize OpenAI when the app starts
@app.on_event("startup")
async def startup_event():
    """Initialize OpenAI when the app starts"""
    global llm
    
    # Check if API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("Please add your OpenAI API key to the .env file")
        return
    
    try:
        # Initialize the ChatGPT model
        llm = ChatOpenAI(
            temperature=0.7,  # Controls creativity (0-1)
            model="gpt-4o"    # Use GPT-4
        )
        print("✅ OpenAI initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing OpenAI: {e}")
```

**🧠 What's happening here?**
- `temperature=0.7`: Makes responses creative but not too random
- `gpt-4o`: Uses the powerful GPT-4 model
- We initialize once when the app starts for efficiency

### Step 8: Creating the Chat Endpoint

Now let's create an endpoint where users can ask questions!

#### 8.1 Define Request/Response Models

Add these after your imports:

```python
# Define what a chat request looks like
class ChatRequest(BaseModel):
    question: str  # The user's question

# Define what a chat response looks like  
class ChatResponse(BaseModel):
    answer: str    # The AI's answer
    sources: list = []  # Sources (empty for now)
```

**🏗️ Why Pydantic models?** They automatically validate data and generate API documentation.

#### 8.2 Create the Chat Endpoint

Add this endpoint:

```python
@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Let users chat with AI"""
    
    # Check if AI is initialized
    if llm is None:
        raise HTTPException(
            status_code=500, 
            detail="AI not initialized. Check your OpenAI API key."
        )
    
    try:
        # Ask the AI a question
        response = llm.invoke([
            {"role": "user", "content": request.question}
        ])
        
        # Return the response
        return ChatResponse(
            answer=response.content,
            sources=[]  # We'll add document sources later
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting AI response: {str(e)}"
        )
```

### Step 9: Testing AI Responses

#### 9.1 Update Your HTML Interface

Add a simple chat interface to your HTML:

```html
<!-- Add this inside your <body> tag -->
<div class="chat-section">
    <h3>💬 Chat with AI</h3>
    <div style="margin: 20px 0;">
        <textarea 
            id="questionInput" 
            placeholder="Ask me anything..."
            style="width: 100%; height: 100px; padding: 10px;"
        ></textarea>
        <br>
        <button 
            onclick="askQuestion()"
            style="background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer;"
        >
            Ask Question
        </button>
    </div>
    <div id="response" style="margin-top: 20px; padding: 15px; background: #f0f0f0;"></div>
</div>

<script>
async function askQuestion() {
    const question = document.getElementById('questionInput').value;
    const responseDiv = document.getElementById('response');
    
    if (!question.trim()) {
        responseDiv.innerHTML = '<p style="color: red;">Please enter a question!</p>';
        return;
    }
    
    responseDiv.innerHTML = '<p>🤔 Thinking...</p>';
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            responseDiv.innerHTML = `
                <h4>Question: ${question}</h4>
                <p><strong>Answer:</strong> ${data.answer}</p>
            `;
        } else {
            responseDiv.innerHTML = `<p style="color: red;">Error: ${data.detail}</p>`;
        }
    } catch (error) {
        responseDiv.innerHTML = '<p style="color: red;">Error connecting to server!</p>';
    }
    
    document.getElementById('questionInput').value = '';
}
</script>
```

#### 9.2 Test Your AI Chat

1. Restart your app: `python main.py`
2. Go to `http://localhost:8000`
3. Type a question like "What is Python?"
4. Click "Ask Question"

**🎉 Amazing!** You're now chatting with AI!

---

## Part 4: Document Processing (90 minutes)

### Step 10: File Upload Functionality

Now let's add the ability to upload documents!

#### 10.1 Add File Upload Imports

Add these imports to your `main.py`:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile
import PyPDF2
import docx
```

#### 10.2 Create File Upload Endpoint

Add this endpoint:

```python
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    
    print(f"📁 Received file: {file.filename}")
    
    # Check file type
    supported_types = ['.pdf', '.txt', '.docx']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in supported_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {supported_types}"
        )
    
    print(f"✅ File type {file_extension} is supported")
    
    # We'll process the file in the next step
    return {"message": f"File {file.filename} uploaded successfully!"}
```

#### 10.3 Add Upload Interface to HTML

Add this section to your HTML:

```html
<div class="upload-section">
    <h3>📁 Upload Document</h3>
    <input 
        type="file" 
        id="fileInput" 
        accept=".pdf,.txt,.docx"
        style="margin: 10px 0; padding: 10px;"
    >
    <br>
    <button 
        onclick="uploadFile()"
        style="background: #2196F3; color: white; padding: 10px 20px; border: none; cursor: pointer;"
    >
        Upload Document
    </button>
    <div id="uploadStatus" style="margin-top: 10px;"></div>
</div>

<script>
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const statusDiv = document.getElementById('uploadStatus');
    
    if (!file) {
        statusDiv.innerHTML = '<p style="color: red;">Please select a file first!</p>';
        return;
    }
    
    statusDiv.innerHTML = '<p>⏳ Uploading...</p>';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            statusDiv.innerHTML = `<p style="color: green;">✅ ${data.message}</p>`;
        } else {
            statusDiv.innerHTML = `<p style="color: red;">❌ Error: ${data.detail}</p>`;
        }
    } catch (error) {
        statusDiv.innerHTML = '<p style="color: red;">❌ Upload failed!</p>';
    }
}
</script>
```

### Step 11: Text Extraction from Documents

Now let's extract text from different file types.

#### 11.1 Create Text Extraction Function

Add this function to your `main.py`:

```python
def extract_text_from_file(file_path: str, filename: str, file_extension: str):
    """Extract text from different file types"""
    
    print(f"📖 Extracting text from {file_extension} file")
    
    try:
        if file_extension == '.pdf':
            # Handle PDF files
            print("📕 Processing PDF...")
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    print(f"✅ Extracted text from page {page_num + 1}")
                
                return text
                
        elif file_extension == '.txt':
            # Handle text files
            print("📄 Processing text file...")
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                print(f"✅ Read {len(text)} characters")
                return text
                
        elif file_extension == '.docx':
            # Handle Word documents
            print("📘 Processing Word document...")
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            print(f"✅ Extracted text from {len(doc.paragraphs)} paragraphs")
            return text
            
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        raise Exception(f"Could not extract text from {file_extension} file: {str(e)}")
```

#### 11.2 Update Upload Endpoint

Update your upload endpoint to actually process files:

```python
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    
    print(f"📁 Received file: {file.filename}")
    
    # Check file type
    supported_types = ['.pdf', '.txt', '.docx']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in supported_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {supported_types}"
        )
    
    tmp_file_path = None
    
    try:
        # Save uploaded file temporarily
        print("💾 Saving file temporarily...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        print(f"✅ File saved to: {tmp_file_path}")
        
        # Extract text from the file
        text = extract_text_from_file(tmp_file_path, file.filename, file_extension)
        
        if not text.strip():
            raise Exception("No text content found in the document")
        
        print(f"✅ Extracted {len(text)} characters of text")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        print("🧹 Cleaned up temporary file")
        
        return {
            "message": f"Successfully processed {file.filename}",
            "text_length": len(text),
            "preview": text[:200] + "..." if len(text) > 200 else text
        }
        
    except Exception as e:
        # Clean up temporary file if error occurs
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        print(f"❌ Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
```

### Step 12: Document Chunking

Large documents need to be split into smaller chunks for better AI processing.

#### 12.1 Add Text Splitting Import

Add this import:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
```

#### 12.2 Create Document Chunking Function

Add this function:

```python
def create_document_chunks(text: str, filename: str):
    """Split document text into smaller chunks"""
    
    print("✂️ Splitting document into chunks...")
    
    # Create a document object
    document = Document(
        page_content=text,
        metadata={"source": filename}
    )
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Maximum characters per chunk
        chunk_overlap=200,      # Overlap between chunks to maintain context
        length_function=len,    # How to measure chunk size
        separators=["\n\n", "\n", " ", ""]  # How to split text
    )
    
    # Split the document
    chunks = text_splitter.split_documents([document])
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show chunk information
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"📄 Chunk {i+1}: {len(chunk.page_content)} characters")
    
    return chunks
```

#### 12.3 Update Upload Endpoint with Chunking

Update your upload endpoint:

```python
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    
    # ... (previous code for file validation and text extraction)
    
    try:
        # ... (file saving and text extraction code)
        
        # Create document chunks
        chunks = create_document_chunks(text, file.filename)
        
        # Store chunks for later use (we'll improve this in the next section)
        global document_chunks
        document_chunks = chunks  # Simple storage for now
        
        return {
            "message": f"Successfully processed {file.filename}",
            "chunks": len(chunks),
            "text_length": len(text)
        }
        
    except Exception as e:
        # ... (error handling code)
```

Add this global variable at the top:

```python
# Global variables
llm = None
document_chunks = []  # Store document chunks temporarily
```

**🧠 Why chunking?** AI models have input limits. Chunking allows us to process large documents efficiently.

---

## Part 5: Vector Database & RAG (90 minutes)

### Step 13: Setting Up ChromaDB

Now we'll add a vector database to store and search document embeddings!

#### 13.1 Add Vector Database Imports

Add these imports:

```python
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
```

#### 13.2 Initialize ChromaDB

Add this setup code:

```python
# Disable ChromaDB telemetry for cleaner logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Global variables for our AI components
llm = None
embeddings = None
vectorstore = None
qa_chain = None

# Initialize ChromaDB client
chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False  # In-memory database for this tutorial
    )
)
```

#### 13.3 Update Startup Event

Update your startup function:

```python
@app.on_event("startup")
async def startup_event():
    """Initialize AI components when the app starts"""
    global llm, embeddings
    
    print("🚀 Starting DocuChat...")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return
    
    try:
        # Initialize OpenAI components
        print("🧠 Initializing AI components...")
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o")
        print("✅ AI components initialized!")
        
        # Test ChromaDB connection
        collections = chroma_client.list_collections()
        print(f"✅ ChromaDB connected! Found {len(collections)} collections")
        
    except Exception as e:
        print(f"❌ Error initializing AI: {e}")
```

### Step 14: Creating Embeddings

Now let's convert our document chunks into embeddings (mathematical representations).

#### 14.1 Update Upload Endpoint with Vector Storage

Replace your upload endpoint with this enhanced version:

```python
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for Q&A"""
    
    print(f"📁 Starting upload: {file.filename}")
    
    # Check if AI components are ready
    if embeddings is None:
        raise HTTPException(
            status_code=500, 
            detail="AI not initialized. Check your OpenAI API key."
        )
    
    # ... (file validation and text extraction code - same as before)
    
    try:
        # ... (file processing code)
        
        # Create document chunks
        chunks = create_document_chunks(text, file.filename)
        
        if not chunks:
            raise Exception("Could not create chunks from document")
        
        # Create vector store from chunks
        print("🔗 Creating vector embeddings...")
        global vectorstore, qa_chain
        
        try:
            if vectorstore is None:
                # Create new vector store
                print("🆕 Creating new vector store...")
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name="documents",
                    embedding_function=embeddings
                )
                # Add documents to the empty vectorstore
                vectorstore.add_documents(chunks)
                print("✅ Vector store created!")
            else:
                # Add to existing vector store
                print("➕ Adding to existing vector store...")
                vectorstore.add_documents(chunks)
                print("✅ Documents added!")
                
        except Exception as ve:
            print(f"❌ Vector store error: {ve}")
            # Try resetting and creating new
            chroma_client.reset()
            vectorstore = Chroma(
                client=chroma_client,
                collection_name="documents",
                embedding_function=embeddings
            )
            vectorstore.add_documents(chunks)
            print("✅ Vector store created after reset!")
        
        # Create Q&A chain
        print("🤖 Creating Q&A system...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Simple strategy: stuff all context into prompt
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Return top 3 matches
            return_source_documents=True  # Include source information
        )
        print("✅ Q&A system ready!")
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return {
            "message": f"Successfully processed {file.filename}",
            "chunks": len(chunks)
        }
        
    except Exception as e:
        # ... (error handling)
```

**🧠 What's happening?**
- **Embeddings**: Convert text to numbers that represent meaning
- **Vector Store**: Database optimized for similarity search
- **Retriever**: Finds relevant chunks based on questions

### Step 15: Building the RAG System

RAG (Retrieval-Augmented Generation) combines information retrieval with AI generation.

#### 15.1 Update Chat Endpoint for Document Q&A

Replace your chat endpoint:

```python
@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """Ask questions about uploaded documents"""
    
    # Check if Q&A system is ready
    if qa_chain is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded yet. Please upload documents first."
        )
    
    try:
        print(f"💬 Processing question: {request.question}")
        
        # Use the Q&A chain to get answer from documents
        result = qa_chain.invoke({"query": request.question})
        
        # Extract source information
        sources = []
        if "source_documents" in result:
            sources = [
                doc.metadata.get("source", "Unknown") 
                for doc in result["source_documents"]
            ]
        
        print(f"✅ Answer generated with {len(sources)} sources")
        
        return ChatResponse(
            answer=result["result"],
            sources=list(set(sources))  # Remove duplicates
        )
        
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )
```

#### 15.2 Test Your RAG System

1. Restart your app
2. Upload a document (PDF, TXT, or DOCX)
3. Ask questions about the document content
4. Notice how the AI now answers based on your specific document!

**🎉 Congratulations!** You've built a complete RAG system!

### Step 15.5: Prompt Engineering for Better Results

Now let's add **prompt engineering** to make our AI responses much better!

#### 15.5.1 Understanding Prompt Engineering

**What is prompt engineering?** It's the art of crafting instructions that guide AI models to give better, more consistent responses.

**Why is it important?** Without good prompts, AI models might:
- Hallucinate (make up information)
- Give vague or unhelpful answers
- Ignore the context documents
- Be inconsistent in their responses

#### 15.5.2 Create a Custom Prompt Template

Add this import to your `main.py`:

```python
from langchain.prompts import PromptTemplate
```

Now, let's create a much better prompt for our Q&A system:

```python
# Define a custom prompt template for better results
prompt_template = """
You are an expert document analysis assistant. Your job is to answer questions based ONLY on the provided context documents.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY the information provided in the context above
2. If the context doesn't contain enough information to answer the question, say "I cannot find enough information in the provided documents to answer this question"
3. Be specific and cite relevant parts of the documents when possible
4. If you're unsure about something, express that uncertainty
5. Keep your answer concise but comprehensive
6. Focus on facts and avoid speculation

Answer:
"""

# Create the prompt template
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
```

#### 15.5.3 Update Your Q&A Chain

Replace your simple Q&A chain with this enhanced version:

```python
# Create/update the Q&A chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}  # Use our custom prompt
)
```

#### 15.5.4 Test Different Prompt Strategies

Let's create multiple prompt templates for different use cases:

```python
# Academic/Research Prompt
academic_prompt = """
You are a research assistant analyzing academic documents. Provide detailed, well-sourced answers.

Context: {context}
Question: {question}

Guidelines:
- Cite specific sections or page numbers when available
- Distinguish between facts and interpretations
- If multiple perspectives exist, present them
- Use academic language but remain accessible

Answer:
"""

# Business/Executive Summary Prompt
business_prompt = """
You are a business analyst providing executive summaries. Focus on key insights and actionable information.

Context: {context}
Question: {question}

Guidelines:
- Lead with the most important information
- Use bullet points for clarity
- Highlight risks, opportunities, and recommendations
- Keep language professional but concise

Answer:
"""

# Creative/Exploratory Prompt
creative_prompt = """
You are a creative research assistant. Help users explore ideas and connections in their documents.

Context: {context}
Question: {question}

Guidelines:
- Think outside the box and make connections
- Suggest related topics or questions
- Use analogies to explain complex concepts
- Encourage further exploration

Answer:
"""
```

#### 15.5.5 Add Prompt Selection to Your App

Add this endpoint to let users choose different prompt styles:

```python
@app.post("/chat/{prompt_style}")
async def chat_with_style(prompt_style: str, request: ChatRequest):
    """Chat with different prompt styles"""
    
    # Define available prompt styles
    prompt_templates = {
        "academic": academic_prompt,
        "business": business_prompt,
        "creative": creative_prompt,
        "default": prompt_template  # Our original prompt
    }
    
    if prompt_style not in prompt_templates:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid prompt style. Available: {list(prompt_templates.keys())}"
        )
    
    # Create a custom chain with the selected prompt
    custom_prompt = PromptTemplate(
        template=prompt_templates[prompt_style],
        input_variables=["context", "question"]
    )
    
    custom_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    
    # Get the response
    result = custom_chain.invoke({"query": request.question})
    
    return ChatResponse(
        answer=result["result"],
        sources=[doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
    )
```

#### 15.5.6 Advanced Prompt Engineering Techniques

Here are some advanced techniques you can teach:

**1. Few-Shot Learning:**
```python
few_shot_prompt = """
You are an expert at analyzing documents. Here are some examples of good answers:

Example 1:
Question: "What is the main conclusion?"
Context: "The study found that 85% of participants showed improvement..."
Good Answer: "Based on the study results, the main conclusion is that 85% of participants showed improvement, indicating the treatment was effective."

Example 2:
Question: "What are the limitations?"
Context: "The study was limited by a small sample size of 50 participants..."
Good Answer: "The study has several limitations, including a small sample size of only 50 participants, which may limit the generalizability of the results."

Now answer this question:
Context: {context}
Question: {question}

Answer:
"""
```

**2. Chain of Thought Prompting:**
```python
chain_of_thought_prompt = """
Analyze this step by step:

Context: {context}
Question: {question}

Step 1: What information is directly relevant to this question?
Step 2: What are the key facts or data points?
Step 3: How do these facts relate to the question?
Step 4: What is the most logical conclusion?

Final Answer:
"""
```

**3. Role-Based Prompting:**
```python
role_based_prompt = """
You are a {role} analyzing this document. Your expertise is in {domain}.

Context: {context}
Question: {question}

As a {role}, what would you focus on? What insights would you provide?

Answer:
"""
```

#### 15.5.7 Testing Your Prompt Engineering

Test different prompts with the same question:

1. **Upload a document** (e.g., a research paper)
2. **Ask the same question** using different prompt styles:
   - `/chat/default` - Standard prompt
   - `/chat/academic` - Academic analysis
   - `/chat/business` - Executive summary
   - `/chat/creative` - Exploratory analysis

**Notice the differences in:**
- Tone and style
- Level of detail
- Focus areas
- Citation style

#### 15.5.8 Prompt Engineering Best Practices

**✅ Do:**
- Be specific about the task
- Provide clear instructions
- Give examples when helpful
- Set boundaries and constraints
- Test and iterate

**❌ Don't:**
- Use vague instructions
- Make prompts too long
- Forget to test with real data
- Ignore edge cases
- Use one-size-fits-all prompts

**🎯 Key Takeaway:** Good prompt engineering can dramatically improve AI performance. It's a crucial skill for building effective AI applications!

---

## Part 6: Advanced Features & Polish (75 minutes)

### Step 16: Enhanced Error Handling

Let's make our app more robust with better error handling.

#### 16.1 Update Health Check

Enhance your health check to show more information:

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    
    # Check if template file exists (we'll create this next)
    template_exists = os.path.exists(
        os.path.join(os.path.dirname(__file__), "templates", "index.html")
    )
    
    return {
        "status": "healthy",
        "message": "DocuChat is running!",
        "components": {
            "openai_embeddings": embeddings is not None,
            "openai_llm": llm is not None,
            "vector_store": vectorstore is not None,
            "qa_chain": qa_chain is not None,
            "template_file": template_exists
        }
    }
```

#### 16.2 Add Better Error Messages

Update your endpoints with more helpful error messages and logging.

### Step 17: Beautiful Web Interface

Let's create a professional-looking web interface!

#### 17.1 Create Professional HTML Template

Create `templates/index.html` with this beautiful interface:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocuChat - Smart Document Q&A</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            color: #4CAF50;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        
        .upload-area {
            border: 3px dashed #4CAF50;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8fffe;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            background: #f0fff0;
            border-color: #45a049;
        }
        
        .btn {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            margin: 10px 5px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
        
        .chat-container {
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .message {
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
            max-width: 80%;
        }
        
        .message.user {
            background: #4CAF50;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .message.bot {
            background: white;
            border: 1px solid #e0e0e0;
            margin-right: auto;
        }
        
        .footer {
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 DocuChat</h1>
            <p>Upload documents and chat with them using AI</p>
        </div>
        
        <div class="content">
            <!-- Upload Section -->
            <div class="section">
                <h2>📁 Upload Document</h2>
                <div class="upload-area" onclick="document.getElementById('file-input').click()">
                    <div style="font-size: 3em; margin-bottom: 10px;">📄</div>
                    <p><strong>Click to upload</strong> or drag and drop</p>
                    <p>Supports PDF, TXT, and DOCX files</p>
                </div>
                <input type="file" id="file-input" accept=".pdf,.txt,.docx" style="display: none;">
                <button class="btn" id="upload-btn" disabled>Upload Document</button>
                <div id="upload-status"></div>
            </div>
            
            <!-- Chat Section -->
            <div class="section">
                <h2>💬 Chat with Your Documents</h2>
                <div class="chat-container">
                    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                        <input 
                            type="text" 
                            id="question-input" 
                            placeholder="Ask a question about your documents..." 
                            disabled
                            style="flex: 1; padding: 12px; border: 2px solid #e0e0e0; border-radius: 25px;"
                        >
                        <button class="btn" id="ask-btn" disabled>Ask</button>
                    </div>
                    <div id="chat-messages"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Built with FastAPI, LangChain, and OpenAI • DocuChat Tutorial</p>
        </div>
    </div>

    <script>
        // JavaScript for file upload and chat functionality
        // (We'll add the JavaScript here)
    </script>
</body>
</html>
```

#### 17.2 Update main.py to Use Template File

Update your root endpoint to read from the template file:

```python
@app.get("/")
async def root():
    """Serve the main page from template file"""
    try:
        # Read HTML content from template file
        html_file_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback if template is missing
        error_html = """
        <html>
            <body style="text-align: center; margin-top: 100px;">
                <h1>Template Error</h1>
                <p>The template file 'templates/index.html' was not found.</p>
                <p>Please make sure the templates directory exists.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)
```

### Step 18: Final Testing & Deployment

#### 18.1 Complete Testing Checklist

Test each feature systematically:

1. **✅ Basic Functionality**
   - App starts without errors
   - Health check endpoint works
   - Web interface loads

2. **✅ File Upload**
   - PDF files upload and process
   - TXT files upload and process  
   - DOCX files upload and process
   - Error handling for unsupported files

3. **✅ AI Chat**
   - Questions get answered
   - Responses are relevant to documents
   - Source citations are included

4. **✅ Error Handling**
   - Graceful handling of invalid inputs
   - Helpful error messages
   - Recovery from failures

#### 18.2 Performance Optimization Tips

```python
# Add these optimizations to your app:

# 1. Limit file size
@app.post("/upload")
async def upload_document(file: UploadFile = File(..., max_size=10_000_000)):  # 10MB limit
    """Upload with size limit"""
    # ... rest of your code

# 2. Add request timeout
@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """Chat with timeout protection"""
    import asyncio
    
    try:
        # Set a timeout for AI responses
        result = await asyncio.wait_for(
            qa_chain.ainvoke({"query": request.question}),
            timeout=30.0  # 30 second timeout
        )
        # ... rest of your code
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
```

#### 18.3 Production Deployment Preparation

Create a production-ready startup script:

```python
# Add this to your main.py for production
if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for this tutorial
        access_log=True
    )
```

---

## 🎉 Congratulations!

You've successfully built a complete document Q&A system! Here's what you've accomplished:

### ✅ What You've Built
- **Modern API**: FastAPI with automatic documentation
- **AI Integration**: OpenAI GPT-4 for intelligent responses
- **Document Processing**: Support for PDF, TXT, and DOCX files
- **Vector Database**: ChromaDB for efficient document search
- **RAG System**: Retrieval-Augmented Generation for accurate answers
- **Beautiful UI**: Professional web interface
- **Error Handling**: Robust error management

### 🧠 Skills You've Learned
- **FastAPI Development**: Building modern Python APIs
- **LangChain Framework**: Working with AI and documents
- **Vector Databases**: Understanding embeddings and similarity search
- **OpenAI Integration**: Using GPT models effectively
- **Full-Stack Development**: Combining backend and frontend
- **Project Organization**: Clean, maintainable code structure

### 🚀 Next Steps

Now that you have a solid foundation, consider these enhancements:

1. **Database Persistence**: Add PostgreSQL for permanent storage
2. **User Authentication**: Add login and user management
3. **Multiple Collections**: Organize documents by topic
4. **Advanced RAG**: Implement more sophisticated retrieval strategies
5. **API Keys Management**: Add user-specific API key management
6. **Deployment**: Deploy to cloud platforms like Heroku or AWS

### 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **LangChain Documentation**: https://python.langchain.com/
- **OpenAI API Reference**: https://platform.openai.com/docs/
- **ChromaDB Documentation**: https://docs.trychroma.com/

---

## 🤝 Getting Help

If you encounter issues during the tutorial:

1. **Check the logs**: Your terminal output contains helpful debugging information
2. **Verify API keys**: Make sure your OpenAI API key is correct and has credits
3. **Check file formats**: Ensure your test documents are valid PDF, TXT, or DOCX files
4. **Review dependencies**: Make sure all packages installed correctly

Remember: Building AI applications is a journey. Each error is a learning opportunity!

**Happy coding!** 🚀✨