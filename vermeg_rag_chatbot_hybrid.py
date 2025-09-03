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
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove email addresses and URLs
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove copyright notices and contact information
        text = re.sub(r'©.*?(?=\n|$)', '', text)
        text = re.sub(r'(?i)contact.*?(?=\n|$)', '', text)
        
        # Clean up punctuation but preserve sentence structure
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of reasonable size"""
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
        """Extract and process text from PDF"""
        logger.info(f"Processing: {pdf_path}")
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean the text
                        text += " " + self.clean_text(page_text)
                
                # Split into manageable chunks
                if text:
                    chunks = self.split_into_chunks(text)
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
        
        return chunks

    def get_question_type(self, query: str) -> Dict[str, float]:
        """Identify the type of question and its characteristics"""
        query_lower = query.lower()
        
        # Define question type patterns
        patterns = {
            'definition': {
                'keywords': {'what is', 'what are', 'define', 'explain', 'describe', 'tell me about'},
                'weight': 1.5
            },
            'capability': {
                'keywords': {'how can', 'how does', 'what can', 'capabilities', 'features', 'benefits'},
                'weight': 1.3
            },
            'comparison': {
                'keywords': {'compare', 'difference', 'better than', 'versus', 'vs', 'advantages'},
                'weight': 1.4
            },
            'implementation': {
                'keywords': {'how to', 'implement', 'install', 'setup', 'configure', 'use'},
                'weight': 1.2
            },
            'technical': {
                'keywords': {'architecture', 'technology', 'platform', 'system requirements', 'integration'},
                'weight': 1.3
            }
        }
        
        # Score each question type
        type_scores = {}
        for qtype, pattern in patterns.items():
            score = 0
            for keyword in pattern['keywords']:
                if keyword in query_lower:
                    score += pattern['weight']
            if score > 0:
                type_scores[qtype] = score
        
        return type_scores

    def process_documents(self, folder_paths: List[str]):
        """Process all PDF documents"""
        logger.info("Loading documents...")
        
        for folder_path in folder_paths:
            pdf_files = list(Path(folder_path).glob("**/*.pdf"))
            for pdf_path in pdf_files:
                chunks = self.extract_pdf_text(str(pdf_path))
                self.documents.extend([(chunk, pdf_path.name) for chunk in chunks])
        
        logger.info(f"Processed {len(self.documents)} chunks from PDFs")

    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find relevant chunks with intelligent matching based on question type"""
        query_lower = query.lower()
        query_words = [word for word in query_lower.split() if len(word) > 2]
        
        # Get question type and corresponding patterns
        question_types = self.get_question_type(query)
        
        # Define content indicators for different types of information
        content_patterns = {
            'definition': ['is a', 'refers to', 'defined as', 'means', 'consists of'],
            'capability': ['provides', 'enables', 'allows', 'features', 'benefits', 'capabilities'],
            'technical': ['architecture', 'platform', 'technology', 'system', 'integration'],
            'implementation': ['steps', 'process', 'procedure', 'implementation', 'configure'],
            'comparison': ['compared to', 'better', 'advantage', 'differs', 'unique']
        }
        
        scored_chunks = []
        for chunk, source in self.documents:
            chunk_lower = chunk.lower()
            base_score = 0
            
            # Basic word matching with position weighting
            for word in query_words:
                if word in chunk_lower:
                    base_score += 1
                    # Position-based scoring
                    position = chunk_lower.index(word)
                    if position < len(chunk_lower) // 4:
                        base_score += 1.5
                    elif position < len(chunk_lower) // 2:
                        base_score += 0.75
            
            # Exact phrase matching
            if query_lower in chunk_lower:
                base_score += 3.0
            
            # Question type specific scoring
            final_score = base_score
            for qtype, type_score in question_types.items():
                if qtype in content_patterns:
                    # Check for type-specific patterns
                    for pattern in content_patterns[qtype]:
                        if pattern in chunk_lower:
                            final_score += type_score
                            
            # Boost score for structured information
            if any(marker in chunk_lower for marker in [
                '• ', '- ', ': ', 'key features', 'benefits include',
                'advantages of', 'designed to', 'specifically for'
            ]):
                final_score *= 1.25
            
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
        # Format prompt for T5
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

Please provide a detailed response:

        # Tokenize and generate
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=1024,  # Increased for more context
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
        """""
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
        
        # Generate answer using T5
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
