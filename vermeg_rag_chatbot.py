"""
Vermeg RAG Chatbot using Google's Gemini Pro API
"""

import os
import PyPDF2
from pathlib import Path
import google.generativeai as genai
from typing import List, Dict
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VermegRAGChatbot:
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",  # Smaller but effective model
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",  # Fast & efficient embedding model
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.documents = []
        self.embeddings = None
        
        logger.info("Initializing models...")
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_model.to(self.device)
        
        # Load LLM for generation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Create generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        logger.info("Models loaded successfully!")

    def load_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF file"""
        logger.info(f"Loading PDF: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def process_documents(self, folder_paths: List[str]):
        """Process all PDF documents in given folders"""
        logger.info("Processing documents...")
        all_chunks = []
        chunk_sources = []  # Keep track of which file each chunk came from
        
        for folder_path in folder_paths:
            pdf_files = list(Path(folder_path).glob("**/*.pdf"))
            for pdf_path in pdf_files:
                try:
                    text = self.load_pdf(str(pdf_path))
                    chunks = self.chunk_text(text)
                    all_chunks.extend(chunks)
                    chunk_sources.extend([pdf_path.name] * len(chunks))
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {str(e)}")
        
        # Create embeddings for all chunks
        logger.info("Creating embeddings...")
        self.documents = list(zip(all_chunks, chunk_sources))
        self.embeddings = self.embedding_model.encode(
            [chunk for chunk, _ in self.documents],
            convert_to_tensor=True,
            show_progress_bar=True
        )
        logger.info(f"Processed {len(self.documents)} chunks from PDFs")

    def get_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks for the query"""
        # Create query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.embeddings.cpu().numpy()
        )[0]
        
        # Get top k chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            chunk, source = self.documents[idx]
            relevant_chunks.append({
                "text": chunk,
                "source": source,
                "similarity": similarities[idx]
            })
        
        return relevant_chunks

    def format_prompt(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Format prompt with context from relevant chunks"""
        context = "\n\n".join([
            f"From {chunk['source']}:\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        prompt = f"""Based on the following context from Vermeg's documentation, answer the question accurately and concisely. If the information is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def generate_response(self, prompt: str) -> str:
        """Generate response using the LLM"""
        response = self.generator(
            prompt,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )[0]["generated_text"]
        
        return response.strip()

    def answer_question(self, query: str) -> Dict:
        """Answer a question using RAG"""
        # Get relevant chunks
        relevant_chunks = self.get_relevant_chunks(query)
        
        # Format prompt with context
        prompt = self.format_prompt(query, relevant_chunks)
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Format sources
        sources = [
            {
                "document": chunk["source"],
                "relevance": f"{chunk['similarity']:.2f}"
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "answer": response,
            "sources": sources
        }

def main():
    # Initialize chatbot
    chatbot = VermegRAGChatbot()
    
    # Process documents
    pdf_folders = [
        "digital solutions",
        "vermeg core solutions"
    ]
    
    logger.info("Loading and processing PDFs...")
    chatbot.process_documents(pdf_folders)
    
    # Interactive loop
    print("\nVermeg RAG Chatbot initialized! Ask questions about Vermeg's solutions (type 'exit' to quit)")
    print("=" * 80)
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'exit':
            break
            
        try:
            result = chatbot.answer_question(query)
            
            print("\nAnswer:")
            print(textwrap.fill(result["answer"], width=80))
            
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['document']} (relevance: {source['relevance']})")
        
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print("Sorry, I encountered an error while processing your question. Please try again.")

if __name__ == "__main__":
    main()
