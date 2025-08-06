# DocuChat - Smart Document Q&A System

A modern web application that allows users to upload documents and ask questions about them using AI. Built with FastAPI, LangChain, and OpenAI.

## Features

- 📄 **Multiple Document Support**: Upload PDF, TXT, and DOCX files
- 🤖 **AI-Powered Q&A**: Ask questions about your documents using GPT-4
- 🎨 **Modern UI**: Beautiful, responsive web interface
- ⚡ **Fast Processing**: Efficient document chunking and vector storage
- 🔍 **Source Citations**: See which documents provided the answers

## Project Structure

```
FastAPILangchain/
├── main.py              # Main FastAPI application
├── templates/
│   └── index.html       # Web interface template
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
└── README.md           # This file
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure OpenAI API

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Run the Application

```bash
python main.py
```

The application will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```
Returns the application status and component readiness.

### Main Interface
```
GET /
```
Serves the web interface for document upload and Q&A.

### Upload Document
```
POST /upload
```
Upload a document (PDF, TXT, or DOCX) for processing.

### Ask Question
```
POST /chat
```
Ask a question about uploaded documents.

**Request body:**
```json
{
  "question": "Your question about the documents"
}
```

**Response:**
```json
{
  "answer": "AI-generated answer",
  "sources": ["document1.pdf", "document2.txt"]
}
```

## Technology Stack

- **Backend**: FastAPI (Python web framework)
- **AI/ML**: 
  - LangChain (AI application framework)
  - OpenAI GPT-4 (Language model)
  - OpenAI Embeddings (Text embeddings)
- **Vector Database**: ChromaDB (In-memory vector storage)
- **Document Processing**: 
  - PyPDF2 (PDF processing)
  - python-docx (Word document processing)
- **Frontend**: HTML, CSS, JavaScript (Vanilla)

## Tutorial Features

This project is designed as a 6-hour tutorial covering:

1. **FastAPI Basics**: API endpoints, file uploads, async operations
2. **LangChain Integration**: Document loaders, text splitters, retrievers
3. **OpenAI API**: Embeddings and chat completions
4. **Vector Databases**: ChromaDB setup and usage
5. **RAG Implementation**: Retrieval-Augmented Generation patterns
6. **Web Development**: Modern HTML/CSS/JS frontend
7. **Project Organization**: Clean code structure and templates

## Development Notes

### Clean Architecture
- HTML templates separated from Python code
- Environment variables for configuration
- Proper error handling and logging
- Modern LangChain patterns (no deprecated methods)

### Educational Focus
- Simple, commented code
- Clear separation of concerns
- Step-by-step progression
- Real-world patterns and best practices

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Make sure your `.env` file exists and contains a valid OpenAI API key
   - Check that the key has sufficient credits

2. **Template Not Found**
   - Ensure the `templates/` directory exists
   - Verify `index.html` is in the templates folder

3. **Module Import Errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Activate your virtual environment

4. **ChromaDB Issues**
   - ChromaDB uses in-memory storage by default
   - Restart the application to clear the vector database

### Development Mode

For development, you can run with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## License

This project is created for educational purposes as part of a LangChain and FastAPI tutorial.