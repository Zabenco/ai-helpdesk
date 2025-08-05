# Universal RAG System

## Features

- **Document RAG (Retrieval-Augmented Generation)** - Searches your knowledge base and generates AI responses based on your documents
- **Local AI Processing** - Uses Ollama with Llama3 for privacy-focused, offline AI responses
- **Conversational Memory** - Maintains chat history for context-aware conversations
- **Override System** - Priority responses for emergencies and special cases
- **REST API** - FastAPI-based web service with automatic documentation
- **Multi-user Support** - Separate conversation histories per user
- **Multiple Document Types** - Supports PDF, TXT, MD, and CSV files


## Let's get it started

### Prerequisites

1. **Install Ollama** (https://ollama.ai)
2. **Download AI Models -I use Llama3-**:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```
3. **Python 3.8+** with pip

### Installation

1. **Clone and setup**:
   ```bash
   git clone https://github.com/Zabenco/ai-helpdesk.git
   cd ai_helpdesk
   python -m venv venv
   ```

2. **Activate virtual environment**:
   ```bash
   # Windows
   venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Setup Your Knowledge Base

1. **Add documents** to the `docs/` folder (Currently inside the folder are demo docs, remove them):
   ```
   docs/
   ├── company_policy.pdf
   ├── troubleshooting_guide.txt
   ├── faq.md
   └── procedures.csv
   ```

2. **Have the AI read and index your decs** (Make sure you are inside your virtual environment:
   ```bash
   python app/ingest.py
   ```

### Run the Helpdesk

1. **Start Ollama** (if not already running, also doesn't need to be in a virtual environment):
   ```bash
   ollama serve
   ```

2. **Start the API server**:
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Open the interactive docs**:
   - Visit: `http://localhost:8000/docs`
   - Test your helpdesk right in the browser!

## Usage Examples

### Basic Question
```json
POST /ask
{
  "question": "How do I reset a password?",
  "user_id": "john_doe"
}
```

### Response
```json
{
  "question": "How do I reset a password?",
  "answer": "To reset a password, follow these steps from our IT policy document...",
  "override_used": false,
  "sources": [
    {
      "file_name": "it_policy.pdf",
      "file_path": "/docs/it_policy.pdf"
    }
  ]
}
```

### Follow-up Question (with memory)
```json
POST /ask
{
  "question": "What if that doesn't work?",
  "user_id": "john_doe"
}
```

The AI remembers the context of your previous password reset question!

## Configuration

### Override System

Create emergency or priority responses in `overrides.json`:

```json
{
  "emergency": "EMERGENCY: Call 911 immediately!",
  "password": "For password issues, contact IT at ext. 1234",
  "server down": "Check the server status dashboard first: http://status.company.com"
}
```

When questions contain these keywords, the AI will prioritize these responses while still using document context.

### Supported File Types

- **PDF** - Text extraction from PDF documents
- **TXT** - Plain text files
- **MD** - Markdown documentation
- **CSV** - Comma-separated data files

## API Endpoints

### `POST /ask`
Ask a question to the AI

**Request Body:**
- `question` (string): Your question
- `user_id` (string, optional): User identifier for conversation memory

**Response:**
- `question`: Echo of your question
- `answer`: AI-generated response
- `override_used`: Whether an override was applied
- `sources`: Documents used to generate the response

### `GET /docs`
Interactive API documentation (Swagger UI)

### `GET /redoc`
Alternative API documentation (ReDoc)

## Development

### Adding New Features

1. **New document types**: Modify `ingest.py` to add support for additional file formats, though a good chunk of files are already available
2. **Custom prompts**: Edit the prompt templates in `main.py` depending on your preferences
3. **Advanced overrides**: Enhance `overrides.py` for complex matching logic

### Rebuilding the Index

When you add or modify documents:
```bash
python app/ingest.py
```

## Privacy & Security

- **Local Processing**: All AI processing happens on your machine
- **No Data Transmission**: Your documents never leave your server or machine. Don't worry about big companies using your data for their learning models.
- **CORS Enabled**: Configure `allow_origins` in `main.py` for production
- **Offline Capable**: Works without internet after initial setup, given you have the knowledge base uploaded

## Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[LlamaIndex](https://www.llamaindex.ai/)** - RAG framework
- **[Ollama](https://ollama.ai/)** - Local AI model serving
- **[Pydantic](https://pydantic.dev/)** - Data validation
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI web server

## Troubleshooting

### Common Issues

**"No index loaded" error**:
- Run `python app/ingest.py` to build the search index

**"Connection refused" to Ollama**:
- Start Ollama: `ollama serve`
- Verify models are installed: `ollama list`

**Import errors**:
- Ensure virtual environment is activated
- Install missing packages within the virtual environment: `pip install -r requirements.txt`

**Empty responses**:
- Check that documents exist in `docs/` folder
- Verify document formats are supported

### Performance Tips

- **RAM Usage**: Large document collections require more memory
- **Response Time**: First query may be slow as models load
- **Storage**: Vector index size depends on document count

---

**Built with love, by Ethan Zabenco**
