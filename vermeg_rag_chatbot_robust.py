"""
Vermeg RAG Chatbot - Simplified robust version
"""

import os
from pathlib import Path
import PyPDF2
import re
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VermegChatbot:
    def __init__(self):
        self.documents = []
    
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

    def process_documents(self, folder_paths: List[str]):
        """Process all PDF documents"""
        logger.info("Loading documents...")
        
        for folder_path in folder_paths:
            pdf_files = list(Path(folder_path).glob("**/*.pdf"))
            for pdf_path in pdf_files:
                chunks = self.extract_pdf_text(str(pdf_path))
                self.documents.extend([(chunk, pdf_path.name) for chunk in chunks])
        
        logger.info(f"Processed {len(self.documents)} chunks from PDFs")

    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find relevant chunks with improved matching"""
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
                    if chunk_lower.index(word) < len(chunk_lower) // 3:
                        score += 0.5
            
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

    def answer_question(self, query: str) -> Dict:
        """Answer a question based on relevant chunks"""
        relevant_chunks = self.find_relevant_chunks(query)
        
        if not relevant_chunks:
            return {
                "answer": "I don't have information about that yet. Could you try asking about a different topic?",
                "sources": []
            }
        
        # Combine relevant chunks into a coherent answer
        answer_parts = []
        seen_content = set()
        
        for chunk in relevant_chunks:
            sentences = [s.strip() for s in re.split(r'[.!?]+', chunk['text']) if s.strip()]
            
            for sentence in sentences:
                # Remove analytical phrases
                sentence = re.sub(r'(?i)(based on|according to|in the( provided)? text)', '', sentence)
                sentence = re.sub(r'^\s*[,.:]\s*', '', sentence)
                
                # Generate a content hash to avoid semantic duplicates
                content_hash = ' '.join(sorted(sentence.lower().split()))
                
                if (content_hash not in seen_content and 
                    len(sentence.split()) > 5 and
                    not any(skip in sentence.lower() for skip in [
                        'copyright', 'all rights reserved', 'contact us', 
                        'learn more', 'click here', 'visit our website'
                    ])):
                    answer_parts.append(sentence.strip())
                    seen_content.add(content_hash)
        
        if answer_parts:
            # Join sentences and clean up
            answer = ' '.join(answer_parts)
            answer = re.sub(r'\s+', ' ', answer)
            answer = answer.strip()
            
            # Make response more conversational based on question type
            if query.lower().startswith(('what', 'which')):
                answer = "Vermeg " + answer[0].lower() + answer[1:]
            elif query.lower().startswith(('where', 'when')):
                answer = answer[0].upper() + answer[1:]
            elif query.lower().startswith('how'):
                answer = "They " + answer[0].lower() + answer[1:]
            else:
                answer = answer[0].upper() + answer[1:]
            
            # Add period if missing
            if not answer.endswith(('.', '!', '?')):
                answer += '.'
        else:
            answer = "I couldn't find specific information about that. Try asking about our solutions or services instead."
        
        return {
            "answer": answer,
            "sources": [{
                "document": chunk["source"],
                "relevance_score": chunk["score"]
            } for chunk in relevant_chunks]
        }

def main():
    # Initialize chatbot
    chatbot = VermegChatbot()
    
    # Process documents
    pdf_folders = [
        "digital solutions",
        "vermeg core solutions"
    ]
    
    logger.info("Loading and processing PDFs...")
    chatbot.process_documents(pdf_folders)
    
    # Interactive loop
    print("\nVermeg Chatbot initialized! Ask questions about Vermeg's solutions (type 'exit' to quit)")
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
