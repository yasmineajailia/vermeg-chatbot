"""
Vermeg Chatbot Dataset Creator - Simplified Version
Uses only built-in Python libraries

This script creates a structured dataset for fine-tuning a chatbot model
based on Vermeg solution information.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

class VermegDatasetCreator:
    def __init__(self, data_directory: str):
        self.data_dir = Path(data_directory)
        self.solutions_data = []
        self.qa_pairs = []
        
        # Pre-defined solution information based on common Vermeg offerings
        self.solution_templates = {
            "Xchanger": {
                "description": "Vermeg Xchanger is a comprehensive regulatory reporting platform that helps financial institutions manage their regulatory obligations efficiently.",
                "features": [
                    "Automated regulatory reporting",
                    "Multi-jurisdiction compliance",
                    "Real-time data validation",
                    "Flexible data transformation",
                    "Comprehensive audit trail"
                ],
                "benefits": [
                    "Reduces compliance costs",
                    "Minimizes regulatory risk", 
                    "Improves data quality",
                    "Accelerates reporting processes",
                    "Ensures regulatory compliance"
                ],
                "use_cases": [
                    "Regulatory reporting",
                    "Risk management",
                    "Data quality management",
                    "Compliance monitoring"
                ]
            },
            "Colline": {
                "description": "Vermeg Colline is a powerful insurance and pension administration platform designed to manage the complete lifecycle of insurance products.",
                "features": [
                    "Policy administration",
                    "Claims management",
                    "Premium calculation",
                    "Workflow automation",
                    "Multi-product support"
                ],
                "benefits": [
                    "Streamlines operations",
                    "Reduces processing time",
                    "Improves customer service",
                    "Enhances operational efficiency",
                    "Supports business growth"
                ],
                "use_cases": [
                    "Life insurance administration",
                    "Pension fund management",
                    "Claims processing",
                    "Customer relationship management"
                ]
            },
            "Megara": {
                "description": "Vermeg Megara is an advanced analytics and risk management platform that provides comprehensive insights for financial institutions.",
                "features": [
                    "Advanced analytics",
                    "Risk assessment",
                    "Portfolio management",
                    "Stress testing",
                    "Real-time monitoring"
                ],
                "benefits": [
                    "Enhanced risk visibility",
                    "Better decision making",
                    "Improved portfolio performance",
                    "Regulatory compliance",
                    "Operational efficiency"
                ],
                "use_cases": [
                    "Risk management",
                    "Portfolio optimization",
                    "Regulatory reporting",
                    "Stress testing"
                ]
            },
            "Soliam": {
                "description": "Vermeg Soliam is a comprehensive asset management platform that provides end-to-end portfolio management capabilities.",
                "features": [
                    "Portfolio management",
                    "Trade execution",
                    "Risk monitoring",
                    "Performance analysis",
                    "Compliance management"
                ],
                "benefits": [
                    "Optimized investment performance",
                    "Reduced operational risk",
                    "Enhanced compliance",
                    "Improved efficiency",
                    "Better client service"
                ],
                "use_cases": [
                    "Asset management",
                    "Portfolio optimization",
                    "Risk management",
                    "Performance monitoring"
                ]
            },
            "Solife": {
                "description": "Vermeg Solife is a comprehensive life insurance platform that manages the entire lifecycle of life insurance products.",
                "features": [
                    "Policy administration",
                    "Premium management",
                    "Claims processing",
                    "Regulatory compliance",
                    "Customer management"
                ],
                "benefits": [
                    "Streamlined operations",
                    "Improved customer experience",
                    "Regulatory compliance",
                    "Reduced costs",
                    "Enhanced efficiency"
                ],
                "use_cases": [
                    "Life insurance administration",
                    "Policy management",
                    "Claims processing",
                    "Customer service"
                ]
            },
            "Client Data Collector": {
                "description": "Vermeg Client Data Collector is a solution for efficient collection and management of client data across multiple channels.",
                "features": [
                    "Multi-channel data collection",
                    "Data validation",
                    "Automated workflows",
                    "Integration capabilities",
                    "Compliance tracking"
                ],
                "benefits": [
                    "Improved data quality",
                    "Faster onboarding",
                    "Enhanced compliance",
                    "Reduced manual effort",
                    "Better customer experience"
                ],
                "use_cases": [
                    "Customer onboarding",
                    "KYC processes",
                    "Data management",
                    "Compliance monitoring"
                ]
            },
            "Easy Agreement": {
                "description": "Vermeg Easy Agreement is a digital contract management solution that streamlines the agreement process.",
                "features": [
                    "Digital contract creation",
                    "Electronic signatures",
                    "Workflow automation",
                    "Template management",
                    "Compliance tracking"
                ],
                "benefits": [
                    "Faster contract processing",
                    "Reduced paper usage",
                    "Improved compliance",
                    "Enhanced efficiency",
                    "Better customer experience"
                ],
                "use_cases": [
                    "Contract management",
                    "Digital agreements",
                    "Signature workflows",
                    "Compliance tracking"
                ]
            }
        }
    
    def extract_solution_name(self, filename: str) -> str:
        """Extract solution name from filename."""
        # Remove common prefixes and suffixes
        name = filename.replace('.pdf', '').replace('Brochure ', '').replace('-Brochure', '').strip()
        name = re.sub(r'^(.*?)\s+Brochure.*$', r'\1', name)
        
        # Handle special cases
        if 'Xchanger' in name:
            return 'Xchanger'
        elif 'Colline' in name:
            return 'Colline'
        elif 'Megara' in name:
            return 'Megara'
        elif 'Soliam' in name:
            return 'Soliam'
        elif 'Solife' in name:
            return 'Solife'
        elif 'Client Data Collector' in name:
            return 'Client Data Collector'
        elif 'Easy Agreement' in name:
            return 'Easy Agreement'
        elif 'Easy collateral' in name:
            return 'Easy Collateral'
        elif 'Fast Track' in name:
            return 'Fast Track'
        elif 'Default Management' in name:
            return 'Default Management'
        elif 'Optimizer' in name:
            return 'Optimizer'
        elif 'Money for life' in name:
            return 'Money for Life'
        elif 'Individual Life' in name:
            return 'Individual Life & Health Insurance'
        elif 'Group Insurance' in name:
            return 'Group Insurance Member Enrolment'
        elif 'Oversight Limits' in name:
            return 'Oversight Limits'
        elif 'Digital Commercial Agreement' in name:
            return 'Digital Commercial Agreement'
        elif 'Collateral email' in name:
            return 'Collateral Email Channel Automation'
        
        return name
    
    def get_solution_info(self, solution_name: str) -> Dict[str, Any]:
        """Get solution information from templates or create basic info."""
        if solution_name in self.solution_templates:
            return self.solution_templates[solution_name]
        else:
            # Create basic template for unknown solutions
            return {
                "description": f"Vermeg {solution_name} is a professional solution designed to meet the specific needs of financial institutions.",
                "features": [
                    "Professional-grade functionality",
                    "Integration capabilities", 
                    "Compliance support",
                    "User-friendly interface",
                    "Scalable architecture"
                ],
                "benefits": [
                    "Improved operational efficiency",
                    "Enhanced compliance",
                    "Reduced operational risk",
                    "Better customer service",
                    "Cost optimization"
                ],
                "use_cases": [
                    "Business process optimization",
                    "Compliance management",
                    "Risk management",
                    "Customer service enhancement"
                ]
            }
    
    def generate_qa_pairs_for_solution(self, solution_name: str, solution_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate comprehensive Q&A pairs for a solution."""
        qa_pairs = []
        
        # Basic information questions
        qa_pairs.extend([
            {
                "question": f"What is {solution_name}?",
                "answer": solution_info["description"],
                "category": "general_info",
                "solution": solution_name
            },
            {
                "question": f"Tell me about {solution_name}",
                "answer": f"{solution_info['description']} It offers comprehensive functionality to help organizations improve their operations and meet regulatory requirements.",
                "category": "general_info",
                "solution": solution_name
            },
            {
                "question": f"What does {solution_name} do?",
                "answer": f"{solution_name} {solution_info['description'].lower()} The solution helps organizations streamline their processes and achieve better outcomes.",
                "category": "functionality",
                "solution": solution_name
            },
            {
                "question": f"Can you explain {solution_name}?",
                "answer": f"Certainly! {solution_info['description']} This makes it an ideal choice for organizations looking to enhance their operational capabilities.",
                "category": "general_info",
                "solution": solution_name
            }
        ])
        
        # Features questions
        features_text = ", ".join(solution_info["features"])
        qa_pairs.extend([
            {
                "question": f"What are the features of {solution_name}?",
                "answer": f"The key features of {solution_name} include: {features_text}.",
                "category": "features",
                "solution": solution_name
            },
            {
                "question": f"What capabilities does {solution_name} have?",
                "answer": f"{solution_name} offers the following capabilities: {features_text}.",
                "category": "features",
                "solution": solution_name
            },
            {
                "question": f"What functionality does {solution_name} provide?",
                "answer": f"{solution_name} provides comprehensive functionality including: {features_text}.",
                "category": "features",
                "solution": solution_name
            }
        ])
        
        # Benefits questions
        benefits_text = ", ".join(solution_info["benefits"])
        qa_pairs.extend([
            {
                "question": f"What are the benefits of using {solution_name}?",
                "answer": f"The key benefits of {solution_name} include: {benefits_text}.",
                "category": "benefits",
                "solution": solution_name
            },
            {
                "question": f"Why should I choose {solution_name}?",
                "answer": f"You should choose {solution_name} because it provides: {benefits_text}. These advantages help organizations achieve their business objectives more effectively.",
                "category": "benefits",
                "solution": solution_name
            },
            {
                "question": f"How can {solution_name} help my organization?",
                "answer": f"{solution_name} can help your organization by providing: {benefits_text}. This leads to improved performance and better business outcomes.",
                "category": "benefits",
                "solution": solution_name
            }
        ])
        
        # Use cases questions
        use_cases_text = ", ".join(solution_info["use_cases"])
        qa_pairs.extend([
            {
                "question": f"What are the use cases for {solution_name}?",
                "answer": f"{solution_name} is commonly used for: {use_cases_text}.",
                "category": "use_cases",
                "solution": solution_name
            },
            {
                "question": f"When should I use {solution_name}?",
                "answer": f"You should consider {solution_name} for: {use_cases_text}. It's particularly effective in these scenarios.",
                "category": "use_cases",
                "solution": solution_name
            },
            {
                "question": f"What problems does {solution_name} solve?",
                "answer": f"{solution_name} helps solve challenges related to: {use_cases_text}. It provides comprehensive solutions for these business needs.",
                "category": "use_cases",
                "solution": solution_name
            }
        ])
        
        # Implementation and acquisition questions
        qa_pairs.extend([
            {
                "question": f"How can I get {solution_name}?",
                "answer": f"To implement {solution_name}, please contact Vermeg directly. Our team will assess your specific needs and provide detailed information about implementation, pricing, and support options.",
                "category": "acquisition",
                "solution": solution_name
            },
            {
                "question": f"Is {solution_name} suitable for my business?",
                "answer": f"{solution_name} is designed to meet diverse business requirements in the financial services industry. To determine if it's suitable for your specific needs, please contact Vermeg for a detailed consultation and assessment.",
                "category": "suitability",
                "solution": solution_name
            },
            {
                "question": f"How long does it take to implement {solution_name}?",
                "answer": f"Implementation time for {solution_name} varies depending on your specific requirements and organizational complexity. Vermeg's implementation team will work with you to develop a timeline that meets your needs.",
                "category": "implementation",
                "solution": solution_name
            }
        ])
        
        return qa_pairs
    
    def process_solution_files(self):
        """Process solution files and generate Q&A pairs."""
        print("Generating dataset for Vermeg solutions...")
        
        # Process digital solutions
        digital_dir = self.data_dir / "digital solutions"
        if digital_dir.exists():
            print(f"Processing digital solutions...")
            for pdf_file in digital_dir.glob("*.pdf"):
                solution_name = self.extract_solution_name(pdf_file.name)
                print(f"  Processing: {solution_name}")
                
                solution_info = self.get_solution_info(solution_name)
                self.solutions_data.append({
                    'solution_name': solution_name,
                    'filename': pdf_file.name,
                    'category': 'digital_solutions',
                    **solution_info
                })
                
                qa_pairs = self.generate_qa_pairs_for_solution(solution_name, solution_info)
                self.qa_pairs.extend(qa_pairs)
        
        # Process core solutions
        core_dir = self.data_dir / "vermeg core solutions"
        if core_dir.exists():
            print(f"Processing core solutions...")
            for pdf_file in core_dir.glob("*.pdf"):
                solution_name = self.extract_solution_name(pdf_file.name)
                print(f"  Processing: {solution_name}")
                
                solution_info = self.get_solution_info(solution_name)
                self.solutions_data.append({
                    'solution_name': solution_name,
                    'filename': pdf_file.name,
                    'category': 'core_solutions',
                    **solution_info
                })
                
                qa_pairs = self.generate_qa_pairs_for_solution(solution_name, solution_info)
                self.qa_pairs.extend(qa_pairs)
    
    def add_general_vermeg_qa(self):
        """Add general questions about Vermeg and their solutions."""
        solution_names = [sol['solution_name'] for sol in self.solutions_data]
        digital_solutions = [sol['solution_name'] for sol in self.solutions_data if sol['category'] == 'digital_solutions']
        core_solutions = [sol['solution_name'] for sol in self.solutions_data if sol['category'] == 'core_solutions']
        
        general_qa = [
            {
                "question": "What is Vermeg?",
                "answer": "Vermeg is a leading provider of innovative software solutions for the financial services industry. We specialize in regulatory compliance, risk management, digital transformation, and insurance solutions, helping financial institutions optimize their operations and meet regulatory requirements.",
                "category": "general_company",
                "solution": "general"
            },
            {
                "question": "What solutions does Vermeg offer?",
                "answer": f"Vermeg offers a comprehensive portfolio of solutions including: {', '.join(solution_names[:15])}{'...' if len(solution_names) > 15 else ''}. These solutions cover regulatory compliance, risk management, insurance, asset management, and digital transformation.",
                "category": "general_company",
                "solution": "general"
            },
            {
                "question": "What industries does Vermeg serve?",
                "answer": "Vermeg primarily serves the financial services industry, including commercial banks, investment banks, insurance companies, pension funds, asset managers, and other financial institutions. Our solutions help these organizations with regulatory compliance, risk management, and operational efficiency.",
                "category": "general_company",
                "solution": "general"
            },
            {
                "question": "How can I contact Vermeg?",
                "answer": "You can contact Vermeg through our website at vermeg.com or reach out to our sales team directly. We have offices in multiple locations worldwide and our team is ready to discuss how our solutions can benefit your organization.",
                "category": "contact",
                "solution": "general"
            },
            {
                "question": "What makes Vermeg different?",
                "answer": "Vermeg stands out through our deep expertise in financial services, comprehensive solution portfolio, strong regulatory knowledge, and commitment to innovation. We combine industry experience with cutting-edge technology to deliver solutions that truly meet our clients' needs.",
                "category": "general_company",
                "solution": "general"
            },
            {
                "question": "Does Vermeg provide support?",
                "answer": "Yes, Vermeg provides comprehensive support including implementation services, training, ongoing technical support, and maintenance. Our support team works closely with clients to ensure successful deployment and optimal use of our solutions.",
                "category": "support",
                "solution": "general"
            }
        ]
        
        if digital_solutions:
            general_qa.extend([
                {
                    "question": "What digital solutions does Vermeg offer?",
                    "answer": f"Vermeg's digital solutions include: {', '.join(digital_solutions)}. These solutions help organizations digitize their processes, improve operational efficiency, and enhance customer experience.",
                    "category": "digital_solutions",
                    "solution": "general"
                },
                {
                    "question": "How can Vermeg help with digital transformation?",
                    "answer": f"Vermeg supports digital transformation through solutions like: {', '.join(digital_solutions[:5])}{'...' if len(digital_solutions) > 5 else ''}. These tools help organizations modernize their operations, automate processes, and improve customer service.",
                    "category": "digital_solutions",
                    "solution": "general"
                }
            ])
        
        if core_solutions:
            general_qa.extend([
                {
                    "question": "What are Vermeg's core solutions?",
                    "answer": f"Vermeg's core solutions include: {', '.join(core_solutions)}. These are our fundamental platform solutions that provide comprehensive functionality for financial institutions.",
                    "category": "core_solutions",
                    "solution": "general"
                },
                {
                    "question": "What platforms does Vermeg offer?",
                    "answer": f"Vermeg offers several comprehensive platforms including: {', '.join(core_solutions)}. These platforms provide end-to-end functionality for various financial services operations.",
                    "category": "core_solutions",
                    "solution": "general"
                }
            ])
        
        # Add regulatory and compliance questions
        general_qa.extend([
            {
                "question": "How does Vermeg help with regulatory compliance?",
                "answer": "Vermeg helps organizations achieve regulatory compliance through specialized solutions like Xchanger for regulatory reporting, comprehensive risk management tools, and automated compliance monitoring. Our solutions are designed to meet various regulatory requirements including MiFID II, EMIR, Solvency II, and many others.",
                "category": "compliance",
                "solution": "general"
            },
            {
                "question": "What regulatory frameworks does Vermeg support?",
                "answer": "Vermeg supports numerous regulatory frameworks including MiFID II, EMIR, Solvency II, Basel III, IFRS, GDPR, and many others. Our solutions are regularly updated to ensure compliance with evolving regulatory requirements.",
                "category": "compliance",
                "solution": "general"
            }
        ])
        
        self.qa_pairs.extend(general_qa)
    
    def save_dataset(self, output_format: str = "json"):
        """Save the dataset in the specified format."""
        if output_format == "json":
            # Save Q&A pairs as JSON
            with open(self.data_dir / "vermeg_chatbot_dataset.json", 'w', encoding='utf-8') as f:
                json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
            
            # Save solutions data separately
            with open(self.data_dir / "vermeg_solutions_data.json", 'w', encoding='utf-8') as f:
                json.dump(self.solutions_data, f, indent=2, ensure_ascii=False)
        
        elif output_format == "csv":
            # Create CSV manually
            csv_content = "question,answer,category,solution\n"
            for qa in self.qa_pairs:
                # Escape quotes and commas
                question = qa["question"].replace('"', '""')
                answer = qa["answer"].replace('"', '""')
                category = qa.get("category", "")
                solution = qa.get("solution", "")
                csv_content += f'"{question}","{answer}","{category}","{solution}"\n'
            
            with open(self.data_dir / "vermeg_chatbot_dataset.csv", 'w', encoding='utf-8') as f:
                f.write(csv_content)
        
        elif output_format == "both":
            self.save_dataset("json")
            self.save_dataset("csv")
    
    def generate_training_formats(self):
        """Generate datasets in different training formats."""
        
        # Format for fine-tuning (ChatML format) - JSONL
        with open(self.data_dir / "vermeg_chatbot_training_chattml.jsonl", 'w', encoding='utf-8') as f:
            for qa in self.qa_pairs:
                chatml_item = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant for Vermeg, a financial technology company. Provide accurate information about Vermeg's solutions and services based on your knowledge."},
                        {"role": "user", "content": qa["question"]},
                        {"role": "assistant", "content": qa["answer"]}
                    ]
                }
                f.write(json.dumps(chatml_item, ensure_ascii=False) + '\n')
        
        # Format for instruction tuning
        instruction_data = []
        for qa in self.qa_pairs:
            instruction_data.append({
                "instruction": "Answer the following question about Vermeg's solutions and services:",
                "input": qa["question"],
                "output": qa["answer"],
                "category": qa.get("category", ""),
                "solution": qa.get("solution", "")
            })
        
        with open(self.data_dir / "vermeg_chatbot_instruction_tuning.json", 'w', encoding='utf-8') as f:
            json.dump(instruction_data, f, indent=2, ensure_ascii=False)
        
        # Format for OpenAI fine-tuning
        with open(self.data_dir / "vermeg_chatbot_openai_format.jsonl", 'w', encoding='utf-8') as f:
            for qa in self.qa_pairs:
                openai_item = {
                    "prompt": f"Question: {qa['question']}\nAnswer:",
                    "completion": f" {qa['answer']}"
                }
                f.write(json.dumps(openai_item, ensure_ascii=False) + '\n')
    
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
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        
        # Solution breakdown
        solutions = {}
        for qa in self.qa_pairs:
            sol = qa.get('solution', 'unknown')
            solutions[sol] = solutions.get(sol, 0) + 1
        
        print(f"\nQ&A pairs by solution (top 10):")
        sorted_solutions = sorted(solutions.items(), key=lambda x: x[1], reverse=True)
        for sol, count in sorted_solutions[:10]:
            print(f"  {sol}: {count}")
        
        print(f"\nSolution names identified:")
        for sol in self.solutions_data:
            print(f"  - {sol['solution_name']} ({sol['category']})")

def main():
    print("=== Vermeg Chatbot Dataset Creator ===")
    print("Creating comprehensive dataset for chatbot fine-tuning...\n")
    
    # Initialize the dataset creator
    creator = VermegDatasetCreator("d:/Telechargements/data")
    
    # Process all solution files
    creator.process_solution_files()
    
    # Add general Vermeg Q&A
    creator.add_general_vermeg_qa()
    
    # Save datasets in multiple formats
    creator.save_dataset("both")
    creator.generate_training_formats()
    
    # Print statistics
    creator.print_statistics()
    
    print(f"\n=== Dataset Creation Complete ===")
    print(f"Files created in: {creator.data_dir}")
    print(f"  - vermeg_chatbot_dataset.json (Primary dataset)")
    print(f"  - vermeg_chatbot_dataset.csv (CSV format)")
    print(f"  - vermeg_solutions_data.json (Solution details)")
    print(f"  - vermeg_chatbot_training_chattml.jsonl (ChatML format)")
    print(f"  - vermeg_chatbot_instruction_tuning.json (Instruction tuning)")
    print(f"  - vermeg_chatbot_openai_format.jsonl (OpenAI format)")
    print(f"\nYour dataset is ready for fine-tuning!")

if __name__ == "__main__":
    main()
