import csv
import json
from collections import Counter, defaultdict
import random

def analyze_dataset(filename):
    """Analyze the dataset for issues"""
    print("=== DATASET ANALYSIS ===")
    
    solution_counts = Counter()
    category_counts = Counter()
    duplicates = defaultdict(list)
    total_rows = 0
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            total_rows += 1
            solution_counts[row['solution']] += 1
            category_counts[row['category']] += 1
            
            # Check for duplicate questions
            question_key = row['question'].lower().strip()
            duplicates[question_key].append(idx)
    
    print(f"Total Q&A pairs: {total_rows}")
    
    # Find duplicates
    actual_duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
    print(f"Duplicate questions found: {len(actual_duplicates)}")
    
    print(f"\n=== SOLUTION DISTRIBUTION ===")
    for solution, count in solution_counts.most_common():
        percentage = (count / total_rows) * 100
        print(f"{solution}: {count} pairs ({percentage:.1f}%)")
    
    print(f"\n=== CATEGORY DISTRIBUTION ===")
    for category, count in category_counts.most_common():
        percentage = (count / total_rows) * 100
        print(f"{category}: {count} pairs ({percentage:.1f}%)")
    
    return solution_counts, category_counts, actual_duplicates, total_rows

def clean_dataset(input_filename, output_filename, target_samples_per_solution=50):
    """Clean and balance the dataset"""
    print(f"\n=== CLEANING DATASET ===")
    print(f"Target: {target_samples_per_solution} samples per solution")
    
    # Read all data
    all_data = []
    seen_questions = set()
    
    with open(input_filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question_key = row['question'].lower().strip()
            # Remove exact duplicates
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                all_data.append(row)
    
    print(f"After removing duplicates: {len(all_data)} unique questions")
    
    # Group by solution
    by_solution = defaultdict(list)
    for row in all_data:
        by_solution[row['solution']].append(row)
    
    # Balance the dataset
    balanced_data = []
    for solution, rows in by_solution.items():
        if len(rows) > target_samples_per_solution:
            # Randomly sample if too many
            sampled = random.sample(rows, target_samples_per_solution)
        else:
            sampled = rows
        balanced_data.extend(sampled)
        print(f"{solution}: {len(sampled)} samples")
    
    # Shuffle the final dataset
    random.shuffle(balanced_data)
    
    # Write cleaned dataset
    with open(output_filename, 'w', newline='', encoding='utf-8') as file:
        if balanced_data:
            fieldnames = balanced_data[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(balanced_data)
    
    print(f"\nCleaned dataset saved: {output_filename}")
    print(f"Total samples: {len(balanced_data)}")
    return len(balanced_data)

def create_training_formats(input_filename):
    """Create different training formats"""
    
    # Read cleaned data
    data = []
    with open(input_filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    
    # 1. ChatML format (for models like Llama, Mistral)
    chatml_data = []
    for row in data:
        chatml_data.append({
            "messages": [
                {"role": "user", "content": row['question']},
                {"role": "assistant", "content": row['answer']}
            ]
        })
    
    with open('vermeg_dataset_chatml.jsonl', 'w', encoding='utf-8') as f:
        for item in chatml_data:
            f.write(json.dumps(item) + '\n')
    
    # 2. Instruction format (for Alpaca-style training)
    instruction_data = []
    for row in data:
        instruction_data.append({
            "instruction": "Answer questions about Vermeg solutions and services.",
            "input": row['question'],
            "output": row['answer']
        })
    
    with open('vermeg_dataset_instruction.json', 'w', encoding='utf-8') as f:
        json.dump(instruction_data, f, indent=2, ensure_ascii=False)
    
    # 3. Simple prompt format
    prompt_data = []
    for row in data:
        prompt_data.append({
            "prompt": f"Question: {row['question']}\nAnswer:",
            "completion": f" {row['answer']}"
        })
    
    with open('vermeg_dataset_prompt.jsonl', 'w', encoding='utf-8') as f:
        for item in prompt_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nTraining formats created:")
    print(f"- ChatML format: vermeg_dataset_chatml.jsonl")
    print(f"- Instruction format: vermeg_dataset_instruction.json") 
    print(f"- Prompt format: vermeg_dataset_prompt.jsonl")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Analyze current dataset
    solution_counts, category_counts, duplicates, total = analyze_dataset('vermeg_chatbot_dataset_expanded.csv')
    
    # Clean and balance dataset
    final_count = clean_dataset('vermeg_chatbot_dataset_expanded.csv', 'vermeg_dataset_cleaned.csv', target_samples_per_solution=40)
    
    # Create training formats
    create_training_formats('vermeg_dataset_cleaned.csv')
    
    print(f"\n=== SUMMARY ===")
    print(f"Original dataset: {total} samples")
    print(f"Cleaned dataset: {final_count} samples")
    print(f"Ready for fine-tuning!")
