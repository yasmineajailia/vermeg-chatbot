# VERBOT - Vermeg AI Chatbot

An intelligent RAG (Retrieval-Augmented Generation) chatbot designed to provide information about Vermeg's financial solutions and services. VERBOT uses the Google Gemini 2.0 Flash API for natural language processing and features semantic search to retrieve relevant information from Vermeg's documentation.

## Features

- **🚀 Fast Startup with Caching**: Embedding cache system reduces startup time from 30+ seconds to ~2 seconds
- **💬 Multilingual Support**: Responds to queries in both English and French
- **📄 Document Processing**: Automatically extracts information from PDF documents
- **🔍 Semantic Search**: Uses sentence embeddings to find relevant content
- **🎨 Modern UI**: Clean, responsive interface with VERBOT branding and typing indicators
- **⚡ API Integration**: Exposes functionality through a FastAPI REST interface
- **🐳 Docker Support**: Easy deployment with containerization
- **🔐 Secure Configuration**: Environment variables for API keys with .env support

## Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key
- Git (for version control)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yasmine2365/vermeg-chatbot.git
   cd vermeg-chatbot
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory:
   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```
   
   **Note**: The `.env` file is gitignored for security. Never commit API keys to the repository.

### Running the Chatbot

#### Local Development

1. Start the API server:
   ```bash
   uvicorn vermeg_api:app --host 127.0.0.1 --port 8002 --reload
   ```

2. Access VERBOT in your browser at http://localhost:8002

   **First Run**: The initial startup will process all PDF documents and create embeddings (~30 seconds)
   
   **Subsequent Runs**: Embeddings are loaded from cache in ~2 seconds ⚡

#### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t vermeg-chatbot .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 -e GOOGLE_API_KEY=your_api_key_here vermeg-chatbot
   ```

3. Access the chatbot at http://localhost:8000

## Project Structure

- `vermeg_rag_chatbot_gemini.py`: Core chatbot implementation with document processing, RAG functionality, and embedding cache
- `vermeg_api.py`: FastAPI wrapper for exposing the chatbot as a REST API
- `static/`: Frontend files including HTML, CSS, and JavaScript for VERBOT interface
- `src/`: Logo and branding assets (VERBOT.png)
- `embeddings_cache/`: Cached document embeddings (auto-generated, gitignored)
- `.env`: Environment variables for API keys (create this file, gitignored)
- `Dockerfile`: Container definition for deployment
- `requirements.txt`: Python dependencies including python-dotenv

## Fine-tuning (Optional)

The repo includes tools for creating and fine-tuning models:

- `dataset_creator.py`: Creates training datasets from documentation
- `finetune_vermeg_chatbot.py`: Fine-tuning script for custom models
- See `finetuning_guide.md` for detailed instructions

## API Endpoints

- `GET /`: Main VERBOT chatbot interface
- `POST /ask`: Submit questions to the chatbot
  ```json
  {
    "question": "What solutions does Vermeg offer for insurance?"
  }
  ```
  
  **Response**:
  ```json
  {
    "answer": "Vermeg offers comprehensive insurance solutions including..."
  }
  ```

## Performance Features

### Embedding Cache System

VERBOT includes an intelligent caching system that dramatically improves startup times:

- **Cache Location**: `embeddings_cache/` directory
- **Cache Keys**: MD5 hashes of PDF filenames
- **Cache Validation**: Automatic invalidation when PDFs are modified
- **Storage Format**: Pickle serialization for fast loading
- **Performance**: ~15x faster startup (2s vs 30s)

To clear the cache and regenerate embeddings:
```bash
rm -rf embeddings_cache  # Linux/macOS
rmdir /s embeddings_cache  # Windows
```

## Customization

### Adding New Documents

1. Place PDF documents in either `digital solutions/` or `vermeg core solutions/` folders
2. Delete the cache to force regeneration:
   ```bash
   rm -rf embeddings_cache
   ```
3. Restart the server - new documents will be processed automatically

### Modifying the UI

The frontend interface is in `static/index.html`:
- Update logo: Replace `src/VERBOT.png`
- Modify colors: Edit CSS variables in `<style>` section
- Change branding: Update title and header text


## Technologies Used

- **AI Model**: Google Gemini 2.0 Flash (via [google-generativeai](https://github.com/google/generative-ai-python))
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Web Framework**: FastAPI with Uvicorn
- **PDF Processing**: PyPDF2
- **Environment Management**: python-dotenv
- **Caching**: Pickle serialization

## Troubleshooting

### Server Won't Start - Port Already in Use
```bash
# Windows
netstat -ano | findstr :8002
taskkill /F /PID <PID>

# Linux/macOS
lsof -ti:8002 | xargs kill -9
```

### Chatbot Not Responding
1. Check that you're accessing via `http://localhost:8002` (not opening HTML file directly)
2. Verify API key is set in `.env` file
3. Check terminal logs for error messages

### Model Not Found Error
Ensure the model name in `vermeg_rag_chatbot_gemini.py` is `gemini-2.0-flash` (or another available model from your API)

## Acknowledgments

- Built with [Google Generative AI](https://github.com/google/generative-ai-python)
- Embedding model: [SentenceTransformers](https://www.sbert.net/)
- PDF processing: [PyPDF2](https://pypdf2.readthedocs.io/)
<img width="1014" height="741" alt="image" src="https://github.com/user-attachments/assets/a6cf79ca-588e-4258-84a3-9a9edad34a27" />

