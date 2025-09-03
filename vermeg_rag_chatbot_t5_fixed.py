"""
Vermeg RAG Chatbot using T5
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
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
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

    def format_prompt(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Format prompt with context from relevant chunks"""
        context = "\n\n".join([
            f"Document: {chunk['source']}\nContent: {chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        # Format specifically for T5
        prompt = f"Answer this question using the provided context. If the answer is not in the context, say 'I don't have enough information to answer that question.'\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        return prompt

    def answer_question(self, query: str) -> Dict:
        """Answer a question using RAG with T5"""
        # Get relevant chunks
        relevant_chunks = self.get_relevant_chunks(query)
        
        if not relevant_chunks:
            return {
                "answer": "I could not find any relevant information in the documentation to answer your question. Please try rephrasing or ask a different question.",
                "sources": []
            }
        
        # Format prompt with context
        prompt = self.format_prompt(query, relevant_chunks)
        
        try:
            # Tokenize with proper handling
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.model.device)
            
            # Generate response
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=200,  # Control output length
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
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
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I encountered an error while processing your question. This might be due to the length of the context or memory constraints. Please try asking a more specific question.",
                "sources": []
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
            
            if result["sources"]:
                print("\nSources:")
                for source in result["sources"]:
                    print(f"- {source['document']} (relevance score: {source['relevance_score']})")
        
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print("Sorry, I encountered an error while processing your question. Please try again.")

if __name__ == "__main__":
    main()
