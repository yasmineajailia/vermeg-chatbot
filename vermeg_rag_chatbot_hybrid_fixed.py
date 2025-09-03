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
        text = ' '.join(text.split())
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'©.*?(?=\n|$)', '', text)
        text = re.sub(r'(?i)contact.*?(?=\n|$)', '', text)
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
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
        logger.info("Loading documents...")
        
        for folder_path in folder_paths:
            pdf_files = list(Path(folder_path).glob("**/*.pdf"))
            for pdf_path in pdf_files:
                chunks = self.extract_pdf_text(str(pdf_path))
                self.documents.extend([(chunk, pdf_path.name) for chunk in chunks])
        
        logger.info(f"Processed {len(self.documents)} chunks from PDFs")

    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        # Normalize query and extract key terms
        query_lower = query.lower()
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        
        # Define question-specific keywords based on question type
        location_keywords = ['located', 'office', 'headquarter', 'hq', 'based', 'presence', 'location', 'country', 'region', 'city']
        definition_keywords = ['what is', 'what are', 'define', 'explain', 'describe', 'mean']
        capability_keywords = ['can', 'able', 'capability', 'feature', 'provide', 'offer', 'support']
        
        # Detect question type
        is_location_question = any(word in query_lower for word in location_keywords)
        is_definition_question = any(phrase in query_lower for phrase in definition_keywords)
        is_capability_question = any(word in query_lower for word in capability_keywords)
        
        scored_chunks = []
        
        for chunk, source in self.documents:
            chunk_lower = chunk.lower()
            score = 0
            
            # Basic word matching
            for word in query_words:
                if word in chunk_lower:
                    score += 1
                    position = chunk_lower.index(word)
                    if position < len(chunk_lower) // 4:
                        score += 1.5
                    elif position < len(chunk_lower) // 2:
                        score += 0.75
            
            # Question-type specific scoring
            if is_location_question:
                if any(word in chunk_lower for word in location_keywords):
                    score *= 2.0
                # Look for text patterns that might indicate locations
                if re.search(r'\bin\s+[A-Z][a-zA-Z]+', chunk):  # "in Paris", "in London", etc.
                    score *= 1.5
                
            elif is_definition_question:
                if any(pattern in chunk_lower for pattern in [
                    'is a', 'are a', 'refers to', 'defined as', 'means'
                ]):
                    score *= 2.0
                    
            elif is_capability_question:
                if any(pattern in chunk_lower for pattern in [
                    'provides', 'offers', 'enables', 'supports', 'allows',
                    'capable of', 'features', 'functionality'
                ]):
                    score *= 1.5
            
            # Boost exact phrase matches
            if query_lower in chunk_lower:
                score += 3.0
                
                scored_chunks.append({
                    'text': chunk,
                    'source': source,
                    'score': score
                })
        
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return scored_chunks[:top_k]

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""You are a knowledgeable assistant for Vermeg. Using ONLY the provided information, give a clear, professional, and well-structured answer to the question.

Important guidelines:
1. Focus on DIRECTLY answering the specific question asked
2. If the question asks about locations, focus on geographical information
3. If the question asks for a definition, start with a clear definition
4. If the question asks about capabilities, focus on what the solution can do
5. If the information is not in the provided context, explicitly state: "I don't have specific information about [topic] in the provided documentation."
6. Do not make assumptions or include information not present in the context

Available Information:
{context}

Question: {query}

Please provide a detailed response:"""

        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=300,
            min_length=100,
            num_beams=6,
            length_penalty=1.2,
            repetition_penalty=1.5,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer_question(self, query: str) -> Dict:
        relevant_chunks = self.find_relevant_chunks(query)
        
        if not relevant_chunks:
            return {
                "answer": "I could not find any relevant information to answer your question.",
                "sources": []
            }
        
        contexts = []
        for chunk in relevant_chunks:
            source_name = chunk['source'].replace('.pdf', '')
            context_entry = f"According to {source_name}:\n{chunk['text']}"
            contexts.append(context_entry)
        
        context = "\n\n".join(contexts)
        
        try:
            answer = self.generate_answer(query, context)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = " ".join([chunk['text'] for chunk in relevant_chunks])
        
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
    chatbot = VermegHybridChatbot()
    
    pdf_folders = [
        "digital solutions",
        "vermeg core solutions"
    ]
    
    logger.info("Loading and processing PDFs...")
    chatbot.process_documents(pdf_folders)
    
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
