"""
Dataset Expander for Vermeg Chatbot

This script takes the existing dataset and creates additional question variations
to increase the training data size.
"""

import json
import csv
import random
from typing import List, Dict

class DatasetExpander:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.original_data = []
        self.expanded_data = []
        
        # Question variation templates
        self.question_templates = {
            "general_info": [
                "What is {solution}?",
                "Tell me about {solution}",
                "Can you explain {solution}?",
                "What exactly is {solution}?",
                "I'd like to know about {solution}",
                "Could you describe {solution}?",
                "Give me information about {solution}",
                "What can you tell me about {solution}?",
                "Explain {solution} to me",
                "I need information on {solution}",
                "What's {solution} all about?",
                "Help me understand {solution}",
                "Provide details about {solution}",
                "I'm interested in learning about {solution}",
                "What should I know about {solution}?"
            ],
            "features": [
                "What are the features of {solution}?",
                "What capabilities does {solution} have?",
                "What functionality does {solution} provide?",
                "What can {solution} do?",
                "List the features of {solution}",
                "What are {solution}'s key features?",
                "What functions does {solution} offer?",
                "What are the main capabilities of {solution}?",
                "Tell me about {solution}'s features",
                "What functionality is included in {solution}?",
                "What does {solution} offer in terms of features?",
                "What are the technical capabilities of {solution}?",
                "Describe the features of {solution}",
                "What can I do with {solution}?",
                "What tools does {solution} provide?"
            ],
            "benefits": [
                "What are the benefits of using {solution}?",
                "Why should I choose {solution}?",
                "How can {solution} help my organization?",
                "What advantages does {solution} offer?",
                "Why is {solution} beneficial?",
                "What value does {solution} provide?",
                "How will {solution} benefit my business?",
                "What are the advantages of {solution}?",
                "Why would I want to use {solution}?",
                "What makes {solution} valuable?",
                "How does {solution} add value?",
                "What positive impact can {solution} have?",
                "What are the business benefits of {solution}?",
                "How can {solution} improve my operations?",
                "What ROI can I expect from {solution}?"
            ],
            "use_cases": [
                "What are the use cases for {solution}?",
                "When should I use {solution}?",
                "What problems does {solution} solve?",
                "In what situations is {solution} useful?",
                "What scenarios is {solution} designed for?",
                "When is {solution} most effective?",
                "What business problems can {solution} address?",
                "Where can {solution} be applied?",
                "What challenges does {solution} help with?",
                "In which industries is {solution} used?",
                "What types of organizations use {solution}?",
                "What business needs does {solution} meet?",
                "How is {solution} typically implemented?",
                "What applications does {solution} have?",
                "For what purposes is {solution} designed?"
            ],
            "acquisition": [
                "How can I get {solution}?",
                "How do I obtain {solution}?",
                "What's the process to acquire {solution}?",
                "How can I purchase {solution}?",
                "How do I get started with {solution}?",
                "What are the steps to get {solution}?",
                "How can I access {solution}?",
                "How do I buy {solution}?",
                "What's needed to get {solution}?",
                "How can I implement {solution}?",
                "How do I request {solution}?",
                "What's the procurement process for {solution}?",
                "How can my company get {solution}?",
                "What do I need to do to get {solution}?",
                "How can I start using {solution}?"
            ],
            "suitability": [
                "Is {solution} suitable for my business?",
                "Is {solution} right for my organization?",
                "Would {solution} work for my company?",
                "Is {solution} a good fit for us?",
                "Should my business consider {solution}?",
                "Is {solution} appropriate for my needs?",
                "Would {solution} be suitable for our requirements?",
                "Is {solution} the right choice for us?",
                "Can {solution} meet our needs?",
                "Is {solution} compatible with our business?",
                "Would {solution} fit our organization?",
                "Is {solution} designed for businesses like ours?",
                "Can {solution} work in our environment?",
                "Is {solution} relevant to our industry?",
                "Would {solution} benefit our type of business?"
            ],
            "implementation": [
                "How long does it take to implement {solution}?",
                "What's the implementation timeline for {solution}?",
                "How long is the deployment process for {solution}?",
                "What's the timeframe for implementing {solution}?",
                "How quickly can {solution} be deployed?",
                "What's the implementation duration for {solution}?",
                "How long before {solution} is operational?",
                "What's the setup time for {solution}?",
                "How long does {solution} take to go live?",
                "What's the rollout timeline for {solution}?",
                "How soon can we start using {solution}?",
                "What's the delivery timeframe for {solution}?",
                "How long is the {solution} implementation project?",
                "When can we expect {solution} to be ready?",
                "What's the time to value for {solution}?"
            ],
            "functionality": [
                "What does {solution} do?",
                "How does {solution} work?",
                "What is the purpose of {solution}?",
                "How does {solution} function?",
                "What's the main function of {solution}?",
                "How does {solution} operate?",
                "What role does {solution} play?",
                "What is {solution} designed to do?",
                "How does {solution} help?",
                "What tasks does {solution} perform?",
                "What job does {solution} do?",
                "What's the core function of {solution}?",
                "How does {solution} serve its purpose?",
                "What processes does {solution} handle?",
                "What operations does {solution} support?"
            ]
        }
    
    def load_original_data(self):
        """Load the original dataset."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.original_data = list(reader)
        print(f"Loaded {len(self.original_data)} original entries")
    
    def generate_question_variations(self, solution_name: str, category: str, answer: str) -> List[Dict]:
        """Generate multiple question variations for a solution and category."""
        variations = []
        
        if category in self.question_templates:
            templates = self.question_templates[category]
            # Generate 3-5 additional variations per original question
            selected_templates = random.sample(templates, min(5, len(templates)))
            
            for template in selected_templates:
                new_question = template.format(solution=solution_name)
                variations.append({
                    'question': new_question,
                    'answer': answer,
                    'category': category,
                    'solution': solution_name
                })
        
        return variations
    
    def create_comparative_questions(self):
        """Create questions comparing solutions."""
        comparative_questions = []
        solutions = list(set([item['solution'] for item in self.original_data if item['solution'] != 'general']))
        
        # General comparison questions
        for i, solution in enumerate(solutions[:10]):  # Limit to avoid too many
            comparative_questions.extend([
                {
                    'question': f"How does {solution} compare to other Vermeg solutions?",
                    'answer': f"{solution} is one of Vermeg's specialized solutions with unique capabilities. Each Vermeg solution is designed for specific use cases. To understand how {solution} compares to other solutions for your specific needs, please contact Vermeg for a detailed comparison and consultation.",
                    'category': 'comparison',
                    'solution': solution
                },
                {
                    'question': f"What makes {solution} different from other solutions?",
                    'answer': f"{solution} has distinctive features and capabilities that set it apart. The specific differences depend on your requirements and the solutions you're comparing. Vermeg can provide a detailed analysis of how {solution} differs from other options in the market.",
                    'category': 'comparison',
                    'solution': solution
                },
                {
                    'question': f"Is {solution} better than competing products?",
                    'answer': f"{solution} is designed with Vermeg's expertise in financial services to provide superior functionality and value. The best choice depends on your specific requirements, existing infrastructure, and business objectives. Contact Vermeg for a detailed comparison with your current solutions.",
                    'category': 'comparison',
                    'solution': solution
                }
            ])
        
        return comparative_questions
    
    def create_industry_specific_questions(self):
        """Create industry-specific questions."""
        industries = ['banking', 'insurance', 'asset management', 'pension funds', 'financial institutions']
        industry_questions = []
        
        solutions = list(set([item['solution'] for item in self.original_data if item['solution'] != 'general']))
        
        for solution in solutions[:8]:  # Limit to avoid too many
            for industry in industries[:3]:  # Top 3 industries
                industry_questions.extend([
                    {
                        'question': f"Can {solution} be used in {industry}?",
                        'answer': f"Yes, {solution} is designed for the financial services industry and can be effectively used in {industry}. Vermeg's solutions are built with deep understanding of various financial sectors. Contact Vermeg to discuss how {solution} can be tailored to your specific {industry} requirements.",
                        'category': 'industry_specific',
                        'solution': solution
                    },
                    {
                        'question': f"Is {solution} suitable for {industry} organizations?",
                        'answer': f"{solution} is well-suited for {industry} organizations as it's designed specifically for the financial services sector. Vermeg has extensive experience working with {industry} clients and can customize {solution} to meet sector-specific needs and regulatory requirements.",
                        'category': 'industry_specific',
                        'solution': solution
                    }
                ])
        
        return industry_questions
    
    def create_technical_questions(self):
        """Create technical integration and deployment questions."""
        technical_questions = []
        solutions = list(set([item['solution'] for item in self.original_data if item['solution'] != 'general']))
        
        for solution in solutions[:10]:
            technical_questions.extend([
                {
                    'question': f"What are the technical requirements for {solution}?",
                    'answer': f"The technical requirements for {solution} depend on your specific deployment scenario, data volumes, and integration needs. Vermeg's technical team will work with you to assess your infrastructure and provide detailed technical specifications and requirements for {solution}.",
                    'category': 'technical',
                    'solution': solution
                },
                {
                    'question': f"Can {solution} integrate with our existing systems?",
                    'answer': f"Yes, {solution} is designed with integration capabilities to work with existing financial systems. Vermeg provides comprehensive integration support and can work with your IT team to ensure seamless connectivity with your current infrastructure and applications.",
                    'category': 'technical',
                    'solution': solution
                },
                {
                    'question': f"What support does Vermeg provide for {solution}?",
                    'answer': f"Vermeg provides comprehensive support for {solution} including implementation services, training, ongoing technical support, maintenance, and updates. Our support team has deep expertise in {solution} and is committed to ensuring your success with the platform.",
                    'category': 'support',
                    'solution': solution
                }
            ])
        
        return technical_questions
    
    def expand_dataset(self):
        """Expand the dataset with variations and new questions."""
        print("Expanding dataset...")
        
        # Start with original data
        self.expanded_data = self.original_data.copy()
        
        # Track unique question-answer pairs to avoid duplicates
        existing_questions = set([item['question'].lower() for item in self.original_data])
        
        # Generate variations for existing questions
        for item in self.original_data:
            if item['solution'] != 'general':  # Skip general questions for variations
                variations = self.generate_question_variations(
                    item['solution'], 
                    item['category'], 
                    item['answer']
                )
                
                # Add only unique variations
                for variation in variations:
                    if variation['question'].lower() not in existing_questions:
                        self.expanded_data.append(variation)
                        existing_questions.add(variation['question'].lower())
        
        # Add comparative questions
        comparative_questions = self.create_comparative_questions()
        for q in comparative_questions:
            if q['question'].lower() not in existing_questions:
                self.expanded_data.append(q)
                existing_questions.add(q['question'].lower())
        
        # Add industry-specific questions
        industry_questions = self.create_industry_specific_questions()
        for q in industry_questions:
            if q['question'].lower() not in existing_questions:
                self.expanded_data.append(q)
                existing_questions.add(q['question'].lower())
        
        # Add technical questions
        technical_questions = self.create_technical_questions()
        for q in technical_questions:
            if q['question'].lower() not in existing_questions:
                self.expanded_data.append(q)
                existing_questions.add(q['question'].lower())
        
        print(f"Expanded from {len(self.original_data)} to {len(self.expanded_data)} entries")
        print(f"Added {len(self.expanded_data) - len(self.original_data)} new questions")
    
    def save_expanded_dataset(self):
        """Save the expanded dataset."""
        # Save as CSV
        with open('vermeg_chatbot_dataset_expanded.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['question', 'answer', 'category', 'solution']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.expanded_data)
        
        # Save as JSON
        with open('vermeg_chatbot_dataset_expanded.json', 'w', encoding='utf-8') as f:
            json.dump(self.expanded_data, f, indent=2, ensure_ascii=False)
        
        # Generate training formats
        self.generate_training_formats()
        
        print("Saved expanded dataset files:")
        print("  - vermeg_chatbot_dataset_expanded.csv")
        print("  - vermeg_chatbot_dataset_expanded.json")
        print("  - vermeg_chatbot_expanded_chattml.jsonl")
        print("  - vermeg_chatbot_expanded_instruction.json")
    
    def generate_training_formats(self):
        """Generate training formats for the expanded dataset."""
        # ChatML format
        with open('vermeg_chatbot_expanded_chattml.jsonl', 'w', encoding='utf-8') as f:
            for item in self.expanded_data:
                chatml_item = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant for Vermeg, a financial technology company. Provide accurate information about Vermeg's solutions and services based on your knowledge."},
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]}
                    ]
                }
                f.write(json.dumps(chatml_item, ensure_ascii=False) + '\n')
        
        # Instruction tuning format
        instruction_data = []
        for item in self.expanded_data:
            instruction_data.append({
                "instruction": "Answer the following question about Vermeg's solutions and services:",
                "input": item["question"],
                "output": item["answer"],
                "category": item.get("category", ""),
                "solution": item.get("solution", "")
            })
        
        with open('vermeg_chatbot_expanded_instruction.json', 'w', encoding='utf-8') as f:
            json.dump(instruction_data, f, indent=2, ensure_ascii=False)
    
    def print_statistics(self):
        """Print statistics about the expanded dataset."""
        print(f"\n=== Expanded Dataset Statistics ===")
        print(f"Total Q&A pairs: {len(self.expanded_data)}")
        
        # Category breakdown
        categories = {}
        for item in self.expanded_data:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nQ&A pairs by category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        
        # Solution breakdown
        solutions = {}
        for item in self.expanded_data:
            sol = item.get('solution', 'unknown')
            solutions[sol] = solutions.get(sol, 0) + 1
        
        print(f"\nTop 10 solutions by Q&A count:")
        sorted_solutions = sorted(solutions.items(), key=lambda x: x[1], reverse=True)
        for sol, count in sorted_solutions[:10]:
            print(f"  {sol}: {count}")

def main():
    print("=== Vermeg Dataset Expander ===")
    print("Expanding your dataset for better fine-tuning results...\n")
    
    expander = DatasetExpander('vermeg_chatbot_dataset.csv')
    expander.load_original_data()
    expander.expand_dataset()
    expander.save_expanded_dataset()
    expander.print_statistics()
    
    print(f"\n=== Expansion Complete ===")
    print("Your expanded dataset is now ready for fine-tuning!")
    print("Recommended next steps:")
    print("1. Review the expanded dataset for quality")
    print("2. Use the ChatML format for modern LLM fine-tuning")
    print("3. Consider starting with a smaller model if dataset is still limited")

if __name__ == "__main__":
    main()
