"""
Vermeg RAG Chatbot - Enhanced version with better text processing
"""

import os
from pathlib import Path
import PyPDF2
import re
from typing import List, Dict
import logging
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VermegChatbot:
    def __init__(self):
        self.documents = []
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text with better handling"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove email addresses and URLs
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove page numbers and common PDF artifacts
        text = re.sub(r'\b\d+\s*$', '', text)  # Page numbers at end of lines
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers
        
        # Remove copyright notices and contact information
        text = re.sub(r'©.*?(?=\n|$)', '', text)
        text = re.sub(r'(?i)contact.*?(?=\n|$)', '', text)
        
        # Clean up punctuation but preserve sentence structure
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
        
        # Remove multiple periods and other redundant punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def is_valid_sentence(self, sentence: str) -> bool:
        """Check if a sentence is valid and meaningful"""
        # Remove stopwords for content checking
        content_words = [word.lower() for word in sentence.split() 
                        if word.lower() not in self.stop_words]
        
        return (len(content_words) >= 3 and  # Has enough content words
                len(sentence.split()) <= 100 and  # Not too long
                not re.search(r'^\d+\.?\d*$', sentence) and  # Not just a number
                not any(marker in sentence.lower() 
                       for marker in ['copyright', 'all rights reserved', 'contact us']))

    def extract_pdf_text(self, pdf_path: str) -> List[str]:
        """Extract and process text from PDF with improved handling"""
        logger.info(f"Processing: {pdf_path}")
        sections = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                current_section = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # Clean the text
                        text = self.clean_text(text)
                        
                        # Split into sentences using NLTK
                        sentences = sent_tokenize(text)
                        
                        # Filter and process sentences
                        valid_sentences = []
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if self.is_valid_sentence(sentence):
                                valid_sentences.append(sentence)
                        
                        # Group sentences into meaningful sections
                        if valid_sentences:
                            section_text = ' '.join(valid_sentences)
                            if len(section_text) > 50:  # Minimum section length
                                sections.append(section_text)
                
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
                self.documents.extend([(section, pdf_path.name) for section in sections])
        
        logger.info(f"Processed {len(self.documents)} sections from PDFs")

    def find_relevant_sections(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find relevant sections with improved matching"""
        query_words = set(word.lower() for word in query.split() 
                         if word.lower() not in self.stop_words)
        scored_sections = []
        
        for section, source in self.documents:
            section_lower = section.lower()
            score = 0
            
            # Calculate word matches
            matched_words = set()
            for word in query_words:
                if word in section_lower:
                    score += 1
                    matched_words.add(word)
            
            if score > 0:
                # Boost score based on percentage of query words matched
                coverage = len(matched_words) / len(query_words)
                score *= (1 + coverage)
                
                # Boost score for sections that appear to be definitions or descriptions
                if any(pattern in section_lower for pattern in [
                    'is a', 'are a', 'provides', 'offers', 'solution for', 'platform for'
                ]):
                    score *= 1.5
                
                # Boost score for title matches
                if any(word in section_lower[:100] for word in query_words):
                    score *= 1.2
                
                scored_sections.append({
                    'text': section,
                    'source': source,
                    'score': score
                })
        
        scored_sections.sort(key=lambda x: x['score'], reverse=True)
        return scored_sections[:top_k]

    def format_answer(self, relevant_sections: List[Dict]) -> str:
        """Format the answer in a clear and readable way"""
        if not relevant_sections:
            return "I could not find relevant information to answer your question."
        
        answer_parts = []
        seen_content = set()  # To avoid duplicating information
        
        for section in relevant_sections:
            sentences = sent_tokenize(section['text'])
            
            for sentence in sentences:
                sentence = sentence.strip()
                # Avoid duplicates and very short sentences
                if (sentence not in seen_content and 
                    len(sentence) > 20 and 
                    self.is_valid_sentence(sentence)):
                    answer_parts.append(sentence)
                    seen_content.add(sentence)
        
        if not answer_parts:
            return "Could not extract a clear answer from the relevant sections."
        
        return ' '.join(answer_parts)

    def answer_question(self, query: str) -> Dict:
        """Answer a question with improved response formatting"""
        # Find relevant sections
        relevant_sections = self.find_relevant_sections(query)
        
        # Format the answer
        answer = self.format_answer(relevant_sections)
        
        # Format sources
        sources = [
            {
                "document": section["source"],
                "relevance_score": section["score"]
            }
            for section in relevant_sections
        ]
        
        return {
            "answer": answer,
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
