"""
Vermeg RAG Chatbot using T5 - Simple Version
"""

import os
import PyPDF2
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VermegRAGChatbot:
    def __init__(self):
        self.model_name = "google/flan-t5-large"
        self.chunk_size = 500
        self.documents = []
        
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        logger.info("Model loaded successfully!")

    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        logger.info(f"Loading PDF: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def process_documents(self, folder_paths: List[str]):
        """Process all PDF documents in given folders"""
        logger.info("Processing documents...")
        
        for folder_path in folder_paths:
            pdf_files = list(Path(folder_path).glob("**/*.pdf"))
            for pdf_path in pdf_files:
                try:
                    text = self.load_pdf(str(pdf_path))
                    chunks = self.chunk_text(text)
                    self.documents.extend([(chunk, pdf_path.name) for chunk in chunks])
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {str(e)}")
        
        logger.info(f"Processed {len(self.documents)} chunks from PDFs")

    def get_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based retrieval"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk, source in self.documents:
            # Simple scoring based on keyword matches
            score = sum(1 for word in query_words if word in chunk.lower())
            if score > 0:
                scored_chunks.append({
                    "text": chunk,
                    "source": source,
                    "score": score
                })
        
        # Sort by score and get top_k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def answer_question(self, query: str) -> Dict:
        """Answer a question using RAG"""
        # Get relevant chunks
        relevant_chunks = self.get_relevant_chunks(query)
        
        if not relevant_chunks:
            return {
                "answer": "I could not find any relevant information in the documentation to answer your question. Please try rephrasing or ask a different question.",
                "sources": []
            }
        
        # Prepare context
        context = ""
        for chunk in relevant_chunks:
            context += f"\nFrom {chunk['source']}:\n{chunk['text']}\n"
        
        # Create input text
        input_text = f"Answer this question about Vermeg using the provided context. Question: {query}\n\nContext: {context}\n\nAnswer:"
        
        # Tokenize and generate
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                min_length=30,
                num_beams=4,
                temperature=0.7,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format sources
        sources = [
            {
                "document": chunk["source"],
                "relevance_score": chunk["score"]
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "answer": response.strip(),
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
            print(result["answer"])
            
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['document']} (relevance score: {source['relevance_score']})")
        
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print("Sorry, I encountered an error while processing your question. Please try again.")

if __name__ == "__main__":
    main()
