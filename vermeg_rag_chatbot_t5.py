"""
Vermeg RAG Chatbot using Flan-T5
"""

import os
import PyPDF2
from pathlib import Path
from typing import List, Dict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VermegRAGChatbot:
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",  # Using smaller model for better performance
        chunk_size: int = 500,  # Reduced chunk size
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
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Create generation pipeline with better memory handling
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            model_kwargs={"low_cpu_mem_usage": True}
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
        """Enhanced keyword-based retrieval with better scoring"""
        # Prepare query words and important terms
        query_words = set(query.lower().split())
        important_terms = {'service', 'solution', 'product', 'feature', 'platform', 'software', 'technology', 'vermeg'}
        
        scored_chunks = []
        for chunk, source in self.documents:
            chunk_lower = chunk.lower()
            
            # Calculate base score from word matches
            score = sum(2 if word in important_terms else 1 
                       for word in query_words 
                       if word in chunk_lower)
            
            # Boost score if it contains service-related words
            if any(term in chunk_lower for term in important_terms):
                score *= 1.5
                
            # Boost score if content appears to be a description or list
            if any(marker in chunk_lower for marker in ['provides', 'offers', 'includes', 'features', 'benefits']):
                score *= 1.2
            
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
        # Get the most relevant parts of each chunk
        summarized_chunks = []
        for chunk in relevant_chunks:
            # Split into sentences and get the most relevant ones
            sentences = chunk['text'].split('.')
            relevant_sentences = [s for s in sentences if any(q.lower() in s.lower() for q in query.split())]
            if relevant_sentences:
                summarized_text = '. '.join(relevant_sentences[:3]) + '.'  # Take top 3 relevant sentences
            else:
                summarized_text = '. '.join(sentences[:2]) + '.'  # Take first 2 sentences as fallback
            
            summarized_chunks.append(f"From {chunk['source']}:\n{summarized_text}")
        
        context = "\n\n".join(summarized_chunks)
        
        prompt = f"""Summarize Vermeg's solutions based on this context:
{context}

Question: {query}

Answer: """
        return prompt

    def answer_question(self, query: str) -> Dict:
        """Answer a question using RAG"""
        # Get relevant chunks
        relevant_chunks = self.get_relevant_chunks(query)
        
        if not relevant_chunks:
            return {
                "answer": "I could not find any relevant information in the documentation to answer your question. Please try rephrasing or ask a different question.",
                "sources": []
            }
        
        # Format prompt with context in a structured way
        context_text = ""
        for chunk in relevant_chunks:
            clean_text = ' '.join(chunk['text'].split())  # Clean up whitespace
            source = chunk['source'].replace('.pdf', '')
            context_text += f"\nFrom {source}:\n{clean_text}\n"
        
        prompt = f"Generate a detailed answer about Vermeg's services using this information: {context_text}\n\nQuestion: {query}\n\nDetailed Answer:"
        
        # Generate response using T5
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response with better parameters
        outputs = self.generator(
            inputs["input_ids"],
            max_new_tokens=256,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )[0]["generated_text"]
        
        response = outputs
        
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
