import pickle
from pathlib import Path
import hashlib
import PyPDF2
import logging

logger = logging.getLogger(__name__)

class VermegGeminiChatbot:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(__file__).parent.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.documents = []

    def _get_cache_path(self, pdf_path: Path) -> Path:
        """Generate cache file path based on PDF content hash."""
        with open(pdf_path, 'rb') as file:
            content_hash = hashlib.md5(file.read()).hexdigest()
        return self.cache_dir / f"{pdf_path.stem}_{content_hash}.pkl"

    def _split_text(self, text: str):
        """Split text into chunks for processing."""
        # Implementation of text splitting into chunks
        pass

    def load_documents(self, directory: str):
        """Load and process PDF documents with caching."""
        pdf_files = list(Path(directory).glob("**/*.pdf"))
        
        for pdf_path in pdf_files:
            logger.info(f"Processing: {pdf_path}")
            try:
                cache_path = self._get_cache_path(pdf_path)
                
                # Try to load from cache first
                if cache_path.exists():
                    logger.info(f"Loading cached embeddings for {pdf_path.name}")
                    with open(cache_path, 'rb') as f:
                        doc_data = pickle.load(f)
                        self.documents.append(doc_data)
                        continue
                
                # If not in cache, process the PDF
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ' '.join(page.extract_text() for page in pdf_reader.pages)
                
                chunks = self._split_text(text)
                embeddings = self.embedding_model.encode(chunks)
                
                doc_data = {
                    'source': pdf_path.name,
                    'chunks': chunks,
                    'embeddings': embeddings
                }
                
                # Save to cache
                with open(cache_path, 'wb') as f:
                    pickle.dump(doc_data, f)
                
                self.documents.append(doc_data)
                logger.info(f"Generated and cached embeddings for {pdf_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")