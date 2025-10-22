# Vermeg AI Chatbot

An intelligent RAG (Retrieval-Augmented Generation) chatbot designed to provide information about Vermeg's financial solutions and services. This chatbot uses the Google Gemini API for natural language processing and features semantic search to retrieve relevant information from Vermeg's documentation.

## Features

- **Multilingual Support**: Responds to queries in both English and French
- **Document Processing**: Automatically extracts information from PDF documents
- **Semantic Search**: Uses sentence embeddings to find relevant content
- **Modern UI**: Clean, responsive interface with typing indicators
- **API Integration**: Exposes functionality through a FastAPI REST interface
- **Docker Support**: Easy deployment with containerization

## Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vermeg-chatbot.git
   cd vermeg-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google Gemini API key as an environment variable:
   ```bash
   # Windows
   set GOOGLE_API_KEY=your_api_key_here
   
   # Linux/macOS
   export GOOGLE_API_KEY=your_api_key_here
   ```

### Running the Chatbot

#### Local Development

1. Start the API server:
   ```bash
   uvicorn vermeg_api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Access the chatbot interface in your browser at http://localhost:8000

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

- `vermeg_rag_chatbot_gemini.py`: Core chatbot implementation with document processing and RAG functionality
- `vermeg_api.py`: FastAPI wrapper for exposing the chatbot as a REST API
- `static/`: Frontend files including HTML, CSS, and JavaScript
- `Dockerfile`: Container definition for deployment
- `requirements.txt`: Python dependencies

## Fine-tuning (Optional)

The repo includes tools for creating and fine-tuning models:

- `dataset_creator.py`: Creates training datasets from documentation
- `finetune_vermeg_chatbot.py`: Fine-tuning script for custom models
- See `finetuning_guide.md` for detailed instructions

## API Endpoints

- `GET /`: Main chatbot interface
- `POST /ask`: Submit questions to the chatbot
  ```json
  {
    "question": "What solutions does Vermeg offer for insurance?"
  }
  ```

## Customization

### Adding New Documents

1. Create a folder with your PDF documents
2. Update the chatbot initialization in `vermeg_api.py`:
   ```python
   chatbot.load_documents("your_new_folder")
   ```

### Modifying the UI

The frontend interface is contained in `static/index.html` and can be customized to match your branding requirements.


## Acknowledgments

- Built with [Google Generative AI](https://github.com/google/generative-ai-python)
- Embedding model: [SentenceTransformers](https://www.sbert.net/)
- PDF processing: [PyPDF2](https://pypdf2.readthedocs.io/)
<img width="1014" height="741" alt="image" src="https://github.com/user-attachments/assets/a6cf79ca-588e-4258-84a3-9a9edad34a27" />

