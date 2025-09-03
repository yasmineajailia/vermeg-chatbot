"""
Vermeg Chatbot Dataset Creator

This script processes Vermeg solution brochures and creates a structured dataset
for fine-tuning a chatbot model.
"""

import os
import json
import PyPDF2
import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Any

class VermegDatasetCreator:
    def __init__(self, data_directory: str):
        self.data_dir = Path(data_directory)
        self.solutions_data = []
        self.qa_pairs = []
        
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\:]', ' ', text)
        return text.strip()
    
    def extract_solution_info(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract key information about a solution from the text."""
        solution_name = filename.replace('.pdf', '').replace('Brochure ', '').replace('-Brochure', '').strip()
        
        # Basic info extraction patterns
        features = []
        benefits = []
        
        # Look for common patterns in brochures
        lines = text.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if any(keyword in line.lower() for keyword in ['features', 'capabilities', 'functionality']):
                current_section = "features"
            elif any(keyword in line.lower() for keyword in ['benefits', 'advantages', 'value']):
                current_section = "benefits"
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                if current_section == "features":
                    features.append(line[1:].strip())
                elif current_section == "benefits":
                    benefits.append(line[1:].strip())
        
        return {
            'solution_name': solution_name,
            'filename': filename,
            'full_text': text,
            'features': features,
            'benefits': benefits,
            'text_length': len(text)
        }
    
    def generate_qa_pairs(self, solution_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate question-answer pairs for a solution."""
        qa_pairs = []
        solution_name = solution_info['solution_name']
        text = solution_info['full_text']
        features = solution_info['features']
        benefits = solution_info['benefits']
        
        # Basic information questions
        qa_pairs.extend([
            {
                "question": f"What is {solution_name}?",
                "answer": f"{solution_name} is one of Vermeg's solutions. {text[:300]}...",
                "category": "general_info",
                "solution": solution_name
            },
            {
                "question": f"Tell me about {solution_name}",
                "answer": f"{solution_name} is a Vermeg solution that provides comprehensive functionality. {text[:400]}...",
                "category": "general_info", 
                "solution": solution_name
            },
            {
                "question": f"What does {solution_name} do?",
                "answer": f"{solution_name} {text[:350]}...",
                "category": "functionality",
                "solution": solution_name
            }
        ])
        
        # Features questions
        if features:
            features_text = ". ".join(features[:5])  # Limit to first 5 features
            qa_pairs.extend([
                {
                    "question": f"What are the features of {solution_name}?",
                    "answer": f"The key features of {solution_name} include: {features_text}",
                    "category": "features",
                    "solution": solution_name
                },
                {
                    "question": f"What capabilities does {solution_name} have?",
                    "answer": f"{solution_name} offers the following capabilities: {features_text}",
                    "category": "features",
                    "solution": solution_name
                }
            ])
        
        # Benefits questions
        if benefits:
            benefits_text = ". ".join(benefits[:5])  # Limit to first 5 benefits
            qa_pairs.extend([
                {
                    "question": f"What are the benefits of using {solution_name}?",
                    "answer": f"The benefits of {solution_name} include: {benefits_text}",
                    "category": "benefits",
                    "solution": solution_name
                },
                {
                    "question": f"Why should I choose {solution_name}?",
                    "answer": f"You should choose {solution_name} because: {benefits_text}",
                    "category": "benefits",
                    "solution": solution_name
                }
            ])
        
        # Comparison and general Vermeg questions
        qa_pairs.extend([
            {
                "question": f"Is {solution_name} suitable for my business?",
                "answer": f"{solution_name} is designed to meet various business needs. {text[:200]}... Please contact Vermeg for a detailed assessment of your specific requirements.",
                "category": "suitability",
                "solution": solution_name
            },
            {
                "question": f"How can I get {solution_name}?",
                "answer": f"To get {solution_name}, please contact Vermeg directly. Our team will assess your needs and provide you with detailed information about implementation and pricing.",
                "category": "acquisition",
                "solution": solution_name
            }
        ])
        
        return qa_pairs
    
    def process_all_pdfs(self):
        """Process all PDF files in the data directory."""
        print("Processing Vermeg solution brochures...")
        
        # Process digital solutions
        digital_dir = self.data_dir / "digital solutions"
        if digital_dir.exists():
            print(f"Processing digital solutions from {digital_dir}")
            for pdf_file in digital_dir.glob("*.pdf"):
                print(f"  Processing: {pdf_file.name}")
                text = self.extract_pdf_text(str(pdf_file))
                if text:
                    clean_text = self.clean_text(text)
                    solution_info = self.extract_solution_info(clean_text, pdf_file.name)
                    self.solutions_data.append(solution_info)
                    
                    # Generate Q&A pairs for this solution
                    qa_pairs = self.generate_qa_pairs(solution_info)
                    self.qa_pairs.extend(qa_pairs)
        
        # Process core solutions
        core_dir = self.data_dir / "vermeg core solutions"
        if core_dir.exists():
            print(f"Processing core solutions from {core_dir}")
            for pdf_file in core_dir.glob("*.pdf"):
                print(f"  Processing: {pdf_file.name}")
                text = self.extract_pdf_text(str(pdf_file))
                if text:
                    clean_text = self.clean_text(text)
                    solution_info = self.extract_solution_info(clean_text, pdf_file.name)
                    self.solutions_data.append(solution_info)
                    
                    # Generate Q&A pairs for this solution
                    qa_pairs = self.generate_qa_pairs(solution_info)
                    self.qa_pairs.extend(qa_pairs)
    
    def add_general_vermeg_qa(self):
        """Add general questions about Vermeg and their solutions."""
        solution_names = [sol['solution_name'] for sol in self.solutions_data]
        
        general_qa = [
            {
                "question": "What solutions does Vermeg offer?",
                "answer": f"Vermeg offers a comprehensive portfolio of solutions including: {', '.join(solution_names[:10])}{'...' if len(solution_names) > 10 else ''}. These solutions cover various domains from digital banking to insurance and regulatory compliance.",
                "category": "general_company",
                "solution": "general"
            },
            {
                "question": "What is Vermeg?",
                "answer": "Vermeg is a leading provider of innovative software solutions for the financial services industry. We specialize in regulatory compliance, risk management, digital transformation, and insurance solutions.",
                "category": "general_company",
                "solution": "general"
            },
            {
                "question": "What industries does Vermeg serve?",
                "answer": "Vermeg serves the financial services industry, including banks, insurance companies, asset managers, and other financial institutions. Our solutions help these organizations with regulatory compliance, risk management, and digital transformation.",
                "category": "general_company",
                "solution": "general"
            },
            {
                "question": "How can I contact Vermeg?",
                "answer": "You can contact Vermeg through our website or reach out to our sales team for more information about our solutions and how they can benefit your organization.",
                "category": "contact",
                "solution": "general"
            }
        ]
        
        # Add solution category questions
        digital_solutions = [sol['solution_name'] for sol in self.solutions_data if 'digital solutions' in sol['filename'].lower()]
        core_solutions = [sol['solution_name'] for sol in self.solutions_data if 'core solutions' in sol['filename'].lower()]
        
        if digital_solutions:
            general_qa.append({
                "question": "What digital solutions does Vermeg offer?",
                "answer": f"Vermeg's digital solutions include: {', '.join(digital_solutions)}. These solutions help organizations digitize their processes and improve operational efficiency.",
                "category": "digital_solutions",
                "solution": "general"
            })
        
        if core_solutions:
            general_qa.append({
                "question": "What are Vermeg's core solutions?",
                "answer": f"Vermeg's core solutions include: {', '.join(core_solutions)}. These are our fundamental platform solutions that provide comprehensive functionality for financial institutions.",
                "category": "core_solutions", 
                "solution": "general"
            })
        
        self.qa_pairs.extend(general_qa)
    
    def save_dataset(self, output_format: str = "json"):
        """Save the dataset in the specified format."""
        if output_format == "json":
            # Save as JSON
            with open(self.data_dir / "vermeg_chatbot_dataset.json", 'w', encoding='utf-8') as f:
                json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
            
            # Save solutions data separately
            with open(self.data_dir / "vermeg_solutions_data.json", 'w', encoding='utf-8') as f:
                json.dump(self.solutions_data, f, indent=2, ensure_ascii=False)
        
        elif output_format == "csv":
            # Save as CSV
            df = pd.DataFrame(self.qa_pairs)
            df.to_csv(self.data_dir / "vermeg_chatbot_dataset.csv", index=False, encoding='utf-8')
        
        elif output_format == "both":
            self.save_dataset("json")
            self.save_dataset("csv")
    
    def generate_training_formats(self):
        """Generate datasets in different training formats."""
        
        # Format for fine-tuning (ChatML format)
        chatml_data = []
        for qa in self.qa_pairs:
            chatml_data.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant for Vermeg, a financial technology company. Provide accurate information about Vermeg's solutions and services."},
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ]
            })
        
        with open(self.data_dir / "vermeg_chatbot_training_chattml.jsonl", 'w', encoding='utf-8') as f:
            for item in chatml_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Format for instruction tuning
        instruction_data = []
        for qa in self.qa_pairs:
            instruction_data.append({
                "instruction": "Answer the following question about Vermeg's solutions:",
                "input": qa["question"],
                "output": qa["answer"]
            })
        
        with open(self.data_dir / "vermeg_chatbot_instruction_tuning.json", 'w', encoding='utf-8') as f:
            json.dump(instruction_data, f, indent=2, ensure_ascii=False)
    
    def print_statistics(self):
        """Print dataset statistics."""
        print(f"\n=== Dataset Statistics ===")
        print(f"Total solutions processed: {len(self.solutions_data)}")
        print(f"Total Q&A pairs generated: {len(self.qa_pairs)}")
        
        # Category breakdown
        categories = {}
        for qa in self.qa_pairs:
            cat = qa.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nQ&A pairs by category:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")
        
        # Solution breakdown
        solutions = {}
        for qa in self.qa_pairs:
            sol = qa.get('solution', 'unknown')
            solutions[sol] = solutions.get(sol, 0) + 1
        
        print(f"\nQ&A pairs by solution:")
        for sol, count in sorted(solutions.items()):
            print(f"  {sol}: {count}")

def main():
    # Initialize the dataset creator
    creator = VermegDatasetCreator("d:/Telechargements/data")
    
    # Process all PDFs
    creator.process_all_pdfs()
    
    # Add general Vermeg Q&A
    creator.add_general_vermeg_qa()
    
    # Save datasets
    creator.save_dataset("both")
    creator.generate_training_formats()
    
    # Print statistics
    creator.print_statistics()
    
    print(f"\n=== Dataset Creation Complete ===")
    print(f"Files created:")
    print(f"  - vermeg_chatbot_dataset.json")
    print(f"  - vermeg_chatbot_dataset.csv") 
    print(f"  - vermeg_solutions_data.json")
    print(f"  - vermeg_chatbot_training_chattml.jsonl")
    print(f"  - vermeg_chatbot_instruction_tuning.json")

if __name__ == "__main__":
    main()
