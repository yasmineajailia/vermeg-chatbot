"""
Vermeg RAG Chatbot - Simple Version with better PDF handling
"""

import os
from pathlib import Path
import PyPDF2
import re
from typing import List, Dict, Tuple
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
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
        return text.strip()
    
    def extract_pdf_text(self, pdf_path: str) -> List[str]:
        """Extract text from PDF with better handling"""
        logger.info(f"Processing: {pdf_path}")
        sections = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                current_section = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # Clean and normalize the text
                        text = self.clean_text(text)
                        
                        # Split into meaningful chunks (by paragraphs or sections)
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        
                        for para in paragraphs:
                            # If paragraph is too long, split it into smaller chunks
                            if len(para) > 1000:
                                words = para.split()
                                chunks = [' '.join(words[i:i+200]) for i in range(0, len(words), 150)]
                                current_section.extend(chunks)
                            else:
                                current_section.append(para)
                            
                            # When we have enough text, create a section
                            if len(' '.join(current_section)) >= 500:
                                sections.append(' '.join(current_section))
                                current_section = []
                
                # Add any remaining text
                if current_section:
                    sections.append(' '.join(current_section))
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
        
        return sections

    def process_documents(self, folder_paths: List[str]):
        """Process all PDF documents"""
        logger.info("Loading documents...")
        
        for folder_path in folder_paths:
            pdf_files = list(Path(folder_path).glob("**/*.pdf"))
            for pdf_path in pdf_files:
                sections = self.extract_pdf_text(str(pdf_path))
                # Store sections with their source
                self.documents.extend([(section, pdf_path.name) for section in sections])
        
        logger.info(f"Processed {len(self.documents)} sections from PDFs")

    def find_relevant_sections(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find relevant sections using keyword matching and proximity"""
        query_words = query.lower().split()
        scored_sections = []
        
        for section, source in self.documents:
            section_lower = section.lower()
            
            # Calculate word matches and their positions
            matches = []
            score = 0
            
            for word in query_words:
                if word in section_lower:
                    score += 1
                    pos = section_lower.find(word)
                    matches.append(pos)
            
            if score > 0:
                # Boost score based on word proximity
                if matches:
                    matches.sort()
                    if max(matches) - min(matches) < 100:  # Words are close together
                        score *= 1.5
                
                # Boost score for sections that appear to be descriptions
                if any(marker in section_lower for marker in ['is', 'are', 'provides', 'offers']):
                    score *= 1.2
                
                scored_sections.append({
                    'text': section,
                    'source': source,
                    'score': score
                })
        
        # Sort by score and return top_k
        scored_sections.sort(key=lambda x: x['score'], reverse=True)
        return scored_sections[:top_k]

    def answer_question(self, query: str) -> Dict:
        """Answer a question based on relevant sections"""
        # Find relevant sections
        relevant_sections = self.find_relevant_sections(query)
        
        if not relevant_sections:
            return {
                "answer": "I could not find relevant information to answer your question.",
                "sources": []
            }
        
        # Combine relevant sections into an answer
        answer_parts = []
        for section in relevant_sections:
            # Extract relevant sentences from the section
            sentences = section['text'].split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check if sentence is relevant to the query
                if any(word.lower() in sentence.lower() for word in query.split()):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                answer_parts.append('. '.join(relevant_sentences) + '.')
        
        # Format sources
        sources = [
            {
                "document": section["source"],
                "relevance_score": section["score"]
            }
            for section in relevant_sections
        ]
        
        return {
            "answer": "\n".join(answer_parts) if answer_parts else "Could not find a specific answer in the relevant sections.",
            "sources": sources
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
