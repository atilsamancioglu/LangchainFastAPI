# DocuChat - FastAPI + LangChain Tutorial

A simple document Q&A system for learning FastAPI and LangChain integration.

## Tutorial Overview (6 Hours)

### Phase 1: Basic Setup (Hours 1-2)
- FastAPI basics and project structure
- Pydantic models for request/response
- Simple chat endpoint with OpenAI integration
- Basic HTML frontend

### Phase 2: Document Processing (Hours 3-4)
- File upload handling in FastAPI
- Document parsing (PDF, TXT, DOCX)
- Text chunking with LangChain
- ChromaDB vector store integration

### Phase 3: RAG Implementation (Hours 5-6)
- Retrieval-Augmented Generation setup
- Vector similarity search
- Source attribution
- Error handling and best practices

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

4. **Open Browser**
   Navigate to: http://localhost:8000

## Key Learning Points

### FastAPI Concepts Covered:
- ✅ Basic app setup and routing
- ✅ Pydantic models for validation
- ✅ File uploads with multipart forms
- ✅ Error handling with HTTPException
- ✅ Startup events and global state
- ✅ Static file serving and HTML responses

### LangChain Concepts Covered:
- ✅ Document loaders and text splitters
- ✅ Embeddings and vector stores
- ✅ Retrievers and Q&A chains
- ✅ Source document tracking
- ✅ Chain types and parameters

### Integration Concepts:
- ✅ Async/await patterns
- ✅ Global state management
- ✅ Error propagation
- ✅ File handling and cleanup

## API Endpoints

- `GET /` - Main HTML interface
- `POST /upload` - Upload documents
- `POST /chat` - Ask questions
- `GET /health` - Health check

## Supported File Types

- PDF (.pdf)
- Text files (.txt)
- Word documents (.docx)

## Architecture

```
Frontend (HTML/JS) → FastAPI → LangChain → OpenAI/ChromaDB
```

The application demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline with document upload, processing, and intelligent Q&A capabilities.