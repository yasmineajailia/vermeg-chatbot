"""
Vermeg Chatbot Fine-tuning Script - CPU Optimized
Using smaller models optimized for CPU training
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VermegChatbotCPUTrainer:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
       
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def load_model_and_tokenizer(self):
        
        logger.info(f"Loading CPU-optimized model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with CPU optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  
            low_cpu_mem_usage=True,     
        )
        
        # Move to CPU explicitly
        self.model = self.model.to('cpu')
        
        logger.info("Model and tokenizer loaded successfully for CPU training!")
        
        # Print model size info
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {param_count:,}")
        
    def setup_lora(self):
        """Set up LoRA for efficient fine-tuning on CPU"""
        logger.info("Setting up LoRA configuration for CPU...")
        
        # More aggressive LoRA config for CPU to reduce training time
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Smaller rank for faster CPU training
            lora_alpha=16,  # Reduced alpha
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"] if "gpt" in self.model_name.lower() else ["q_proj", "v_proj"],
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
    def prepare_dataset(self, data_file="vermeg_dataset_instruction.json"):
        """Prepare the dataset for training"""
        logger.info(f"Loading dataset from {data_file}")
        
        # Load the data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format data for training with shorter context for CPU
        formatted_data = []
        for item in data:
            # Simpler format for CPU training
            conversation = f"Question: {item['input']}\nAnswer: {item['output']}<|endoftext|>"
            formatted_data.append({"text": conversation})
        
        # Convert to Hugging Face dataset
        self.dataset = Dataset.from_list(formatted_data)
        
        # Smaller validation split for faster training
        split_dataset = self.dataset.train_test_split(test_size=0.05, seed=42)
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.eval_dataset)}")
        
    def tokenize_dataset(self):
        """Tokenize the dataset with CPU-friendly settings"""
        logger.info("Tokenizing dataset for CPU training...")
        
        def tokenize_function(examples):
            # Tokenize with shorter max length for CPU efficiency
            model_inputs = self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=512,  # Shorter sequences for CPU
                return_tensors=None
            )
            
            # For causal language modeling, labels are the same as input_ids
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        self.train_dataset = self.train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        self.eval_dataset = self.eval_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=self.eval_dataset.column_names
        )
        
        logger.info("Dataset tokenization complete!")
        
    def train(self, output_dir="./vermeg-chatbot-cpu"):
        """Train the model with CPU-optimized settings"""
        logger.info("Starting CPU training...")
        start_time = time.time()
        
        # CPU-optimized training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # Fewer epochs for faster training
            per_device_train_batch_size=4,  # Larger batch for CPU
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,  # Reduce accumulation steps
            warmup_steps=20,  # Fewer warmup steps
            learning_rate=5e-4,  # Higher learning rate for faster convergence
            fp16=False,  # No mixed precision on CPU
            logging_steps=5,
            save_steps=100,
            save_total_limit=2,  # Keep fewer checkpoints
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,  # Disable for CPU
            report_to=None,
            dataloader_num_workers=0  # No multiprocessing for simplicity
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        logger.info("Training started - this will take longer on CPU, please be patient...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training complete! Time taken: {elapsed_time/60:.1f} minutes")
        logger.info(f"Model saved to {output_dir}")
        
    def test_model(self, test_questions=None):
        """Test the trained model with sample questions"""
        if test_questions is None:
            test_questions = [
                "What is Xchanger?",
                "Tell me about Vermeg's solutions",
                "How can Easy Agreement help my business?",
                "What are the benefits of using Colline?"
            ]
        
        logger.info("Testing the trained model...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        for question in test_questions:
            # Format the question
            prompt = f"Question: {question}\nAnswer:"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,  # Shorter responses for CPU
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.replace(prompt, "").strip()
            
            print(f"\nQ: {question}")
            print(f"A: {answer}")
            print("-" * 50)

def main():
    
    print("=== Chatbot CPU Fine-tuning ===")
    
    print("Running on CPU")
    
    
    # Initialize trainer with smaller model for CPU
    trainer = VermegChatbotCPUTrainer(
        model_name="microsoft/DialoGPT-small"  # Small, fast model for CPU
    )
    
    try:
        # Step 1: Load model
        trainer.load_model_and_tokenizer()
        
        # Step 2: Setup LoRA
        trainer.setup_lora()
        
        # Step 3: Prepare dataset
        trainer.prepare_dataset()
        
        # Step 4: Tokenize dataset
        trainer.tokenize_dataset()
        
        # Step 5: Train
        trainer.train()
        
        # Step 6: Test
        trainer.test_model()
        
        print("\n=== CPU Training Complete! ===")
        print("Your Vermeg chatbot is ready!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
