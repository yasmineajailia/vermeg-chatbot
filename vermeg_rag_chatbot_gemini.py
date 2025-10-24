"""
Vermeg RAG Chatbot using Google's Gemini API
"""

import os
import PyPDF2
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from langdetect import detect
from translate import Translator
import pickle
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure translation
MAX_TRANSLATION_RETRIES = 3
TRANSLATION_TIMEOUT = 10  # seconds

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=GOOGLE_API_KEY)

class VermegGeminiChatbot:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        cache_dir: str = "embeddings_cache"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.documents = []
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Using SentenceTransformer for document embedding and similarity search
        logger.info("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Models loaded successfully!")
        
        # Add company info for common queries
        company_info = """Vermeg is a global financial technology company that provides specialized software solutions:

        Main Solutions and Services:
        1. Core Solutions:
           - Colline: Enterprise Collateral Management
           - Megara: Securities Processing and Asset Servicing
           - Soliam: Investment Management and Fund Administration
           - Solife: Life Insurance Management

        2. Digital Solutions:
           - Xchanger: Payment processing and ISO message handling
           - Client Data Collector
           - Collateral Management Solutions:
              * Easy Collateral Platform
              * Easy Collateral for ECB
              * Collateral Email Channel Automation
           - Digital Commercial Agreement
           - Default Management
           - Fast Track
           - Optimizer
           - Oversight Limits

        3. Insurance Solutions:
           - Group Insurance Member Enrollment
           - Individual Life & Health Insurance
           - Money for Life Program

        Key Client Segments:
        - Central Banks (23+ central banks as clients)
        - Banks and Buy-Side Institutions
        - CCPs (Central Counterparty Clearing Houses)
        - CSDs (Central Securities Depositories)
        - Asset Servicers
        - Insurance Companies

        Global Presence:
        - Europe: Paris (France), Luxembourg, Brussels (Belgium), Amsterdam (Netherlands), Madrid (Spain)
        - UK and South Africa: London (UK)
        - Asia Pacific: Shanghai (China), Tokyo (Japan), Singapore, Sydney (Australia)
        - Americas: New York (USA), São Paulo (Brazil)
        - Tunisia: Multiple offices in Tunis
        
        Vermeg Group B.V. serves over 160 clients across 40+ countries, with 30+ years of expertise in financial services technology."""
        self.documents.append({
            'source': 'company_info',
            'chunks': [company_info],
            'embeddings': self.embedding_model.encode([company_info])
        })

    def _get_cache_path(self, pdf_path: Path) -> Path:
        """Generate a cache file path for a given PDF."""
        # Create a hash of the PDF path and its modification time
        pdf_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()
        return self.cache_dir / f"{pdf_hash}.pkl"
    
    def _is_cache_valid(self, pdf_path: Path, cache_path: Path) -> bool:
        """Check if the cache file exists and is newer than the PDF."""
        if not cache_path.exists():
            return False
        
        pdf_mtime = pdf_path.stat().st_mtime
        cache_mtime = cache_path.stat().st_mtime
        return cache_mtime > pdf_mtime
    
    def _save_cache(self, cache_path: Path, data: Dict) -> None:
        """Save document data to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")
    
    def _load_cache(self, cache_path: Path) -> Dict:
        """Load document data from cache."""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return None

    def load_documents(self, directory: str) -> None:
        """Load and process PDF documents from the specified directory."""
        logger.info(f"Loading documents from {directory}")
        pdf_files = list(Path(directory).glob("**/*.pdf"))
        
        for pdf_path in pdf_files:
            cache_path = self._get_cache_path(pdf_path)
            
            # Try to load from cache first
            if self._is_cache_valid(pdf_path, cache_path):
                logger.info(f"Loading from cache: {pdf_path.name}")
                cached_data = self._load_cache(cache_path)
                if cached_data:
                    self.documents.append(cached_data)
                    continue
            
            # Process PDF if cache doesn't exist or is invalid
            try:
                logger.info(f"Processing PDF: {pdf_path.name}")
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    
                    # Split text into chunks with overlap
                    chunks = self._split_text(text)
                    
                    # Generate embeddings
                    embeddings = self.embedding_model.encode(chunks)
                    
                    # Store document information
                    doc_info = {
                        'source': str(pdf_path),
                        'chunks': chunks,
                        'embeddings': embeddings
                    }
                    
                    # Save to cache
                    self._save_cache(cache_path, doc_info)
                    
                    self.documents.append(doc_info)
                    logger.info(f"Processed and cached: {pdf_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
        return chunks

    def _find_relevant_chunks(self, query: str, top_k: int = 3) -> tuple[List[str], List[Dict]]:
        """Find the most relevant chunks for the given query using semantic search.
        Returns: (chunks, citations) where citations include source document info."""
        if not self.documents:
            return [], []

        query_embedding = self.embedding_model.encode(query)
        
        all_similarities = []
        all_chunks = []
        all_sources = []
        
        for doc in self.documents:
            similarities = np.dot(doc['embeddings'], query_embedding)
            source_name = Path(doc['source']).name
            
            for i, sim in enumerate(similarities):
                all_similarities.append(sim)
                all_chunks.append(doc['chunks'][i])
                all_sources.append({
                    'source': source_name,
                    'similarity': float(sim)
                })
        
        # Get indices of top-k most similar chunks
        top_indices = np.argsort(all_similarities)[-top_k:][::-1]
        
        top_chunks = [all_chunks[i] for i in top_indices]
        top_citations = [all_sources[i] for i in top_indices]
        
        # Deduplicate citations by source
        seen_sources = set()
        unique_citations = []
        for citation in top_citations:
            if citation['source'] not in seen_sources:
                seen_sources.add(citation['source'])
                unique_citations.append(citation)
        
        return top_chunks, unique_citations

    def generate_response(self, user_query: str) -> Dict:
        """Generate a response using Gemini based on relevant document chunks.
        Returns: Dict with 'answer' and 'citations' keys."""
        try:
            # Detect language
            lang = detect(user_query)
            is_french = lang == 'fr'
            
            # Translate query to English if it's French
            if is_french:
                translator = Translator(to_lang='en', from_lang='fr')
                user_query_en = translator.translate(user_query)
            else:
                user_query_en = user_query

            # Find relevant chunks with increased number for better context
            relevant_chunks, citations = self._find_relevant_chunks(user_query_en, top_k=5)
            
            if not relevant_chunks:
                response = "I don't have information about that yet. Feel free to ask about our solutions like Colline, Megara, or our other services."
                if is_french:
                    translator = Translator(from_lang='en', to_lang='fr')
                    return {"answer": translator.translate(response), "citations": []}
                return {"answer": response, "citations": []}
            
            # Construct the prompt with improved context and instructions
            context = "\n".join(relevant_chunks)
            prompt = f"""You are Vermeg's friendly and knowledgeable AI assistant. 
            {
                "Vous devez répondre en français de manière professionnelle et conviviale." 
                if is_french else 
                "Please respond in English in a professional and friendly manner."
            }

            Information:
            {context}

            Question: {user_query_en}

            Instructions:
            - Respond naturally as if you're part of Vermeg
            - Focus on what you can tell about our solutions and services
            - Be clear and specific about features and benefits
            - Do not mention limitations in available information
            - Never use phrases like "the text mentions" or "according to"
            - Don't add disclaimers about missing information
            - If you don't have enough information about something, simply don't mention it
            - Keep responses positive and focused on what we offer

            Instructions:
            {
                '''
                - Répondez exclusivement en français
                - Structurez votre réponse de manière claire
                - Utilisez des puces pour lister les fonctionnalités ou services
                - Pour les solutions techniques, expliquez brièvement leurs avantages
                - Gardez un ton professionnel mais accessible
                - Si vous mentionnez un produit, précisez son objectif principal
                - Soyez concis et précis
                - N'utilisez pas de phrases comme "selon" ou "d'après"
                - Si l'information est partielle, précisez-le
                '''
                if is_french else
                '''
                - Respond in clear, structured English
                - Use bullet points for features or services
                - For technical solutions, briefly explain their benefits
                - Keep a professional but friendly tone
                - If mentioning a product, include its key purpose
                - Be concise and focused
                - Don't use phrases like "based on" or "according to"
                - If information is partial, acknowledge it
                '''
            }
            """

            # Generate response with language-specific configuration
            generation_config = genai.types.GenerationConfig(
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                max_output_tokens=2048,
                stop_sequences=["\n\n\n"]
            )

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            answer = response.text.strip()
            
            # Detect response language and translate if needed
            if is_french:
                retries = 0
                while retries < MAX_TRANSLATION_RETRIES:
                    try:
                        response_lang = detect(answer)
                        if response_lang == 'en':
                            translator = Translator(from_lang='en', to_lang='fr', timeout=TRANSLATION_TIMEOUT)
                            translated = translator.translate(answer)
                            if translated and len(translated) > 10:  # Basic validation
                                answer = translated
                                break
                            else:
                                logger.warning(f"Translation attempt {retries + 1} failed, retrying...")
                        else:
                            break  # Already in French
                    except Exception as e:
                        logger.warning(f"Translation error on attempt {retries + 1}: {str(e)}")
                    retries += 1
                    if retries == MAX_TRANSLATION_RETRIES:
                        logger.warning("All translation attempts failed, using original response")
            
            # Clean up response
            answer = re.sub(r'(?i)(based on|according to|in the( provided)? text|I can tell you that|selon|d\'après)', '', answer)
            answer = re.sub(r'(?m)^[*\-•] ', '\n• ', answer)
            answer = re.sub(r'(?m)^•', '\n•', answer)
            answer = re.sub(r'\n{3,}', '\n\n', answer)
            answer = answer[0].upper() + answer[1:] if answer else answer
            if answer and not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            return {"answer": answer, "citations": citations}

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_msg = "Je suis désolé, mais j'ai du mal à traiter cette demande. Pourriez-vous reformuler votre question?" if 'is_french' in locals() and is_french else "I apologize, but I'm having trouble processing that request. Could you try rephrasing your question?"
            return {"answer": error_msg, "citations": []}
    
    def generate_response_stream(self, user_query: str):
        """Generate a streaming response using Gemini.
        Yields: Dict chunks with 'type' ('token' or 'citations') and 'content'."""
        try:
            # Detect language
            lang = detect(user_query)
            is_french = lang == 'fr'
            
            # Translate query to English if it's French
            if is_french:
                translator = Translator(to_lang='en', from_lang='fr')
                user_query_en = translator.translate(user_query)
            else:
                user_query_en = user_query

            # Find relevant chunks
            relevant_chunks, citations = self._find_relevant_chunks(user_query_en, top_k=5)
            
            if not relevant_chunks:
                response = "I don't have information about that yet. Feel free to ask about our solutions like Colline, Megara, or our other services."
                if is_french:
                    translator = Translator(from_lang='en', to_lang='fr')
                    response = translator.translate(response)
                yield {"type": "token", "content": response}
                yield {"type": "citations", "content": []}
                return
            
            # Construct the prompt
            context = "\n".join(relevant_chunks)
            prompt = f"""You are Vermeg's friendly and knowledgeable AI assistant. 
            {
                "Vous devez répondre en français de manière professionnelle et conviviale." 
                if is_french else 
                "Please respond in English in a professional and friendly manner."
            }

            Information:
            {context}

            Question: {user_query_en}

            Instructions:
            - Respond naturally as if you're part of Vermeg
            - Focus on what you can tell about our solutions and services
            - Be clear and specific about features and benefits
            - Do not mention limitations in available information
            - Never use phrases like "the text mentions" or "according to"
            - Don't add disclaimers about missing information
            - If you don't have enough information about something, simply don't mention it
            - Keep responses positive and focused on what we offer
            """

            # Generate streaming response
            generation_config = genai.types.GenerationConfig(
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                max_output_tokens=2048,
            )

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            # Stream tokens
            for chunk in response:
                if chunk.text:
                    yield {"type": "token", "content": chunk.text}
            
            # Send citations at the end
            yield {"type": "citations", "content": citations}

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            error_msg = "I apologize, but I'm having trouble processing that request."
            yield {"type": "token", "content": error_msg}
            yield {"type": "citations", "content": []}

def main():
    # Initialize the chatbot
    chatbot = VermegGeminiChatbot()
    
    # Load documents from both solution directories
    chatbot.load_documents("digital solutions")
    chatbot.load_documents("vermeg core solutions")
    
    # Simple interactive loop
    print("Vermeg Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.generate_response(user_input)
        print(f"\nChatbot: {response['answer']}")
        if response['citations']:
            print(f"Sources: {', '.join([c['source'] for c in response['citations']])}")

if __name__ == "__main__":
    main()
