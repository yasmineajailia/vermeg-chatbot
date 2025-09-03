"""
Vermeg RAG Chatbot - Hybrid version combining PDF extraction with T5 model
"""

import os
from pathlib import Path
import PyPDF2
import re
from typing import List, Dict
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VermegHybridChatbot:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """Initialize the chatbot with model and tokenizer."""
        self.documents = []
        
        # Initialize model and tokenizer
        logger.info(f"Loading {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        logger.info(f"Model loaded successfully on {self.device}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = ' '.join(text.split())
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'©.*?(?=\n|$)', '', text)
        text = re.sub(r'(?i)contact.*?(?=\n|$)', '', text)
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of reasonable size."""
        chunks = []
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk) + '.')
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk) + '.')
        
        return chunks

    def extract_pdf_text(self, pdf_path: str) -> List[str]:
        """Extract and process text from PDF."""
        logger.info(f"Processing: {pdf_path}")
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += " " + self.clean_text(page_text)
                
                if text:
                    chunks = self.split_into_chunks(text)
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
        
        return chunks

    def process_documents(self, folder_paths: List[str]):
        """Process all PDF documents."""
        logger.info("Loading documents...")
        
        for folder_path in folder_paths:
            pdf_files = list(Path(folder_path).glob("**/*.pdf"))
            for pdf_path in pdf_files:
                chunks = self.extract_pdf_text(str(pdf_path))
                self.documents.extend([(chunk, pdf_path.name) for chunk in chunks])
        
        logger.info(f"Processed {len(self.documents)} chunks from PDFs")

    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find relevant chunks with improved matching."""
        query_words = [word.lower() for word in query.split() 
                      if len(word) > 2]  # Ignore very short words
        scored_chunks = []
        
        for chunk, source in self.documents:
            chunk_lower = chunk.lower()
            score = 0
            
            # Calculate word matches with position weighting
            for word in query_words:
                if word in chunk_lower:
                    score += 1
                    # Boost score if word appears near the start of the chunk
                    position = chunk_lower.index(word)
                    if position < len(chunk_lower) // 4:  # Increased emphasis on early mentions
                        score += 1.0
                    elif position < len(chunk_lower) // 2:
                        score += 0.5
                        
                    # Additional score for exact phrase matches
                    if query.lower() in chunk_lower:
                        score += 2.0
            
            if score > 0:
                # Boost score for definition-like content
                if any(pattern in chunk_lower for pattern in [
                    'is a', 'are a', 'provides', 'offers', 'solution for', 
                    'platform for', 'designed to', 'enables'
                ]):
                    score *= 1.5
                
                scored_chunks.append({
                    'text': chunk,
                    'source': source,
                    'score': score
                })
        
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return scored_chunks[:top_k]

    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using the T5 model."""
        # Format prompt for T5 with better instructions
        prompt = f"""You are a knowledgeable assistant for Vermeg. Using the provided information, give a clear, professional, and well-structured answer to the question.

Important guidelines:
1. Be concise but comprehensive
2. Use professional business language
3. Combine information from multiple sources when relevant
4. Structure the answer logically
5. If some aspect is not covered in the provided information, clearly state that

Available Information:
{context}

Question: {query}

Please provide a detailed response:"""

        # Tokenize and generate with improved parameters
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=300,    # Allow for longer answers
            min_length=100,    # Ensure comprehensive responses
            num_beams=6,       # Wider beam search
            length_penalty=1.2, # Slightly favor longer responses
            repetition_penalty=1.5,  # Strongly discourage repetition
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer_question(self, query: str) -> Dict:
        """Answer a question using both retrieval and generation."""
        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(query)
        
        if not relevant_chunks:
            return {
                "answer": "I could not find any relevant information to answer your question.",
                "sources": []
            }
        
        # Prepare context for the model with better structure
        contexts = []
        for chunk in relevant_chunks:
            # Add source context more naturally
            source_name = chunk['source'].replace('.pdf', '')
            context_entry = f"According to {source_name}:\n{chunk['text']}"
            contexts.append(context_entry)
        
        context = "\n\n".join(contexts)
        
        try:
            answer = self.generate_answer(query, context)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            # Fallback to extracted text if generation fails
            answer = " ".join([chunk['text'] for chunk in relevant_chunks])
        
        # Format sources
        sources = [
            {
                "document": chunk["source"],
                "relevance_score": chunk["score"]
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "answer": answer,
            "sources": sources
        }

def main():
    # Initialize chatbot
    chatbot = VermegHybridChatbot()
    
    # Process documents
    pdf_folders = [
        "digital solutions",
        "vermeg core solutions"
    ]
    
    logger.info("Loading and processing PDFs...")
    chatbot.process_documents(pdf_folders)
    
    # Interactive loop
    print("\nVermeg Hybrid Chatbot initialized! Ask questions about Vermeg's solutions (type 'exit' to quit)")
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
